/**
 * OpenCode V2 entrypoint for the delegated-access plugin.
 *
 * # Why this file exists
 *
 * OpenCode 2.x's plugin loader rejects the V1 file's function default export
 * (`Plugin must export a default definition with an id and an effect or setup
 * function`), so upstream has been INERT on this fleet's OpenCode 2 hosts:
 * fetched + cached at startup, then skipped at registration. This module
 * supplies the V2 entrypoint the 2.x loader wants:
 *
 *     import { define } from "@opencode-ai/plugin/v2/promise"
 *     export default define({ id: "opencode-delegated-access", setup })
 *
 * The V1 entrypoint (`src/index.ts`, async-function default) stays untouched
 * and keeps working for OpenCode 1.x hosts.
 *
 * # Mapping the V1 machinery onto the V2 (promise) plugin API
 *
 * The V2 promise context has no SDK `client` and no hook-map; everything the
 * V1 entry did through `client.*` is re-expressed through context domains:
 *
 *   - Permission decision     → `ctx.permission.hook("evaluate", …)` — the
 *     evaluation input carries a mutable `effect` ("allow"|"deny"|"ask");
 *     setting `allow` pre-empts the TUI prompt entirely (the V1
 *     `permission.ask` output.status path, now actually dispatched).
 *   - Classifier call         → `ctx.experimental.generate.text({ prompt,
 *     model })` — one stateless, session-less, tool-less model call. The
 *     ephemeral-session dance (create/abort/delete, tools deny-map,
 *     `experimental.chat.system.transform` isolation) is gone: generate.text
 *     has no session, no history and no tools, so the inherited-preamble
 *     failure mode cannot exist. The endpoint has no `system` field, so the
 *     classifier system prompt is folded into `prompt`.
 *   - Root-session walk       → `ctx.session.get({ sessionID })` via
 *     parentID (same fail-closed contract as ui/session-tree.ts).
 *   - Recent user messages    → `ctx.session.context({ sessionID })` — the
 *     post-last-compaction context messages, which is exactly the classifier
 *     context the plugin wants (V1 read all messages incl. pre-compaction).
 *     V2 user messages carry `text` directly and no `agent` field, so the
 *     rootAgent provenance filter is dropped; the strip-tag sanitizer is
 *     kept as defense-in-depth.
 *   - Post-ask resolution     → `ctx.permission.reply({ sessionID, requestID,
 *     decision })` behind the RISKY/failure notification buttons.
 *   - Reply/history events    → `ctx.event.subscribe({ types:
 *     ["permission.asked", "permission.replied"] })`; V2 replied shapes are
 *     normalised to the canonical { sessionID, permissionID, response } the
 *     shared `handlePermissionReplied` already consumes.
 *   - Repo context (git/gh)   → `childProcessShell()` shim standing in for
 *     the Bun `$` the V2 context doesn't provide (see v2-shell.ts).
 *   - Logging                 → console with the "[delegated-access]"
 *     prefix (fire-and-forget, same degradation semantics as log.ts; the V2
 *     promise context has no `client.app.log`).
 *
 * # Deliberate behaviour deltas vs V1 (documented — do not "fix" silently)
 *
 *   1. SAFE verdicts set `effect = "allow"` during evaluation: auto-approval
 *      is pre-prompt, so there is no live TUI prompt to cancel during the
 *      old `safeCountdownMs` window. The safe notification is therefore
 *      informational ("auto-approved"), and `safeCountdownMs` no longer gates
 *      approval timing on this path. All hard-RISKY categories still require
 *      the TUI prompt.
 *   2. The session-model auto-detection uses the latest assistant message
 *      from the root session's context; the authoritative source remains
 *      `options.classifierModel` (our fleet deployment always sets it).
 *   3. The safe-notification batching (macOS single-notification window)
 *      is not ported; Linux notifier behaviour differs and the approval no
 *      longer races the notification.
 */

import { define } from "@opencode-ai/plugin/v2/promise"
import { parseConfig, type DelegatedAccessConfig } from "./config.ts"
import {
  CLASSIFIER_SYSTEM_PROMPT,
  buildClassifierUserPrompt,
  DIRECTORY_CLASSIFIER_SYSTEM_PROMPT,
  buildDirectoryClassifierUserPrompt,
} from "./classifier/prompt.ts"
import { parseVerdict, type Verdict } from "./classifier/parse.ts"
import { resolveClassifierModel, type ModelRef } from "./classifier/model.ts"
import { sanitizeUserMessageForClassifier } from "./ui/messages.ts"
import {
  handlePermissionReplied,
} from "./index.ts"
import { ApprovalHistoryStore } from "./permission/approval-history.ts"
import { DirectoryVerdictCache } from "./permission/directory-cache.ts"
import { PendingSubjectsMap } from "./permission/pending-subjects.ts"
import {
  FailureNotifyRateLimiter,
} from "./permission/failure-notify.ts"
import {
  isDualRepoContext,
  RepoContextCache,
  type DualRepoContext,
  type RepoContext,
} from "./repo-context.ts"
import { SessionRepoContext } from "./session-repo-context.ts"
import { sendNotification } from "./notify/notify.ts"
import { childProcessShell } from "./v2-shell.ts"

// ---------------------------------------------------------------------------
// Minimal structural types for the V2 promise plugin context we consume.
// Deliberately NOT imported from @opencode-ai/plugin: the npm dist ships a
// partially-populated promise surface (1.19.5 lacks permission/session
// domain typings) and the host passes the real object at runtime anyway.
// ---------------------------------------------------------------------------

export type PermissionEffect = "allow" | "deny" | "ask"

export type PermissionEvaluationLike = {
  sessionID: string
  action: string
  resources: ReadonlyArray<string>
  metadata?: Record<string, unknown>
  source?: unknown
  effect: PermissionEffect
  message?: string
}

export type RegistrationLike = { dispose: () => Promise<void> | void }

export type V2PermissionDomain = {
  hook: (
    name: "evaluate",
    callback: (input: PermissionEvaluationLike) => Promise<void> | void,
  ) => Promise<RegistrationLike>
  reply: (input: {
    sessionID: string
    requestID: string
    decision: "once" | "always" | "reject"
    message?: string
  }) => Promise<unknown>
}

export type SessionMessageLike = {
  id?: string
  type?: string
  role?: string
  text?: unknown
  agent?: unknown
  model?: { providerID?: unknown; id?: unknown } | unknown
  providerID?: unknown
  modelID?: unknown
  content?: ReadonlyArray<{ type?: unknown; text?: unknown }>
  parts?: ReadonlyArray<{ type?: unknown; text?: unknown }>
}

export type V2SessionDomain = {
  get: (input: { sessionID: string }) => Promise<unknown>
  context: (input: { sessionID: string }) => Promise<unknown>
}

export type V2EventDomain = {
  subscribe: (options?: {
    types?: ReadonlyArray<string>
    signal?: AbortSignal
  }) => AsyncIterable<unknown>
}

export type GenerateDomainLike = {
  text: (input: {
    prompt: string
    model?: { providerID: string; id: string }
  }) => Promise<unknown>
}

export type V2PluginContext = {
  location?: { directory?: string } | null
  options?: unknown
  permission?: V2PermissionDomain
  session?: V2SessionDomain
  event?: V2EventDomain
  experimental?: { generate?: GenerateDomainLike }
  generate?: GenerateDomainLike
}

// ---------------------------------------------------------------------------
// Logging (console-based; the V2 promise context has no client.app.log).
// ---------------------------------------------------------------------------

export type Logger = {
  debug: (message: string, extra?: Record<string, unknown>) => void
  info: (message: string, extra?: Record<string, unknown>) => void
  warn: (message: string, extra?: Record<string, unknown>) => void
  error: (message: string, extra?: Record<string, unknown>) => void
}

const CONSOLE_PREFIX = "[delegated-access]"

function consoleLogger(): Logger {
  const DEBUG = process.env.DELEGATED_ACCESS_V2_DEBUG === "1"
  const fmt = (
    level: string,
    message: string,
    extra?: Record<string, unknown>,
  ) => {
    const extraPart = extra ? ` ${JSON.stringify(extra)}` : ""
    return `${CONSOLE_PREFIX} ${level} ${message}${extraPart}`
  }
  return {
    debug: (m, e) => {
      if (DEBUG) console.info(fmt("debug", m, e))
    },
    info: (m, e) => console.info(fmt("info", m, e)),
    warn: (m, e) => console.warn(fmt("warn", m, e)),
    error: (m, e) => console.error(fmt("error", m, e)),
  }
}

// ---------------------------------------------------------------------------
// Permission-type matching (mirror of handler.ts's defensive sets).
// ---------------------------------------------------------------------------

/** Runtime permission actions this plugin classifies for bash commands. */
export const BASH_ACTION_MATCHES = new Set(["bash", "command"])
/** Runtime action string for external-directory access. */
export const EXTERNAL_DIRECTORY_ACTION = "external_directory"

/** Usage info the classifier + handler flow needs about one evaluation. */
type EvaluationInfo = {
  sessionID: string
  action: string
  patterns: string[]
  subjectLabel: "command" | "path"
  /** Extracted classifiable subject (full bash command or directory path). */
  command: string | null
}

/**
 * Map a V2 evaluation into the V1-ish classification vocabulary. Re-joins
 * multi-segment bash resources exactly like V1's extractBashCommand.
 */
export function evaluationToInfo(ev: PermissionEvaluationLike): EvaluationInfo {
  const patterns =
    Array.isArray(ev.resources) &&
    ev.resources.every((r): r is string => typeof r === "string")
      ? [...ev.resources]
      : []

  const isBash = BASH_ACTION_MATCHES.has(ev.action)
  const isDirectory = ev.action === EXTERNAL_DIRECTORY_ACTION
  let command: string | null = null
  if (patterns.length === 1) command = patterns[0] ?? null
  else if (patterns.length > 1) {
    command = isBash ? patterns.join(" && ") : (patterns.find(Boolean) ?? null)
  }

  return {
    sessionID: ev.sessionID,
    action: ev.action,
    patterns,
    subjectLabel: isDirectory ? "path" : "command",
    command,
  }
}

/**
 * Normalise a `permission.replied` event payload into the canonical shape
 * `handlePermissionReplied` consumes. V2 field names (requestID/reply) map
 * in; V1 names still win if both appear.
 */
export function normalizeV2RepliedProperties(
  raw: unknown,
): { sessionID: string; permissionID: string; response: string } | null {
  if (!raw || typeof raw !== "object") return null
  const r = raw as Record<string, unknown>
  if (typeof r.sessionID !== "string") return null

  const permissionID =
    typeof r.permissionID === "string"
      ? r.permissionID
      : typeof r.requestID === "string"
        ? r.requestID
        : null
  const response =
    typeof r.response === "string"
      ? r.response
      : typeof r.reply === "string"
        ? r.reply
        : null
  if (permissionID === null || response === null) return null

  return {
    sessionID: r.sessionID,
    permissionID,
    response,
  }
}

// ---------------------------------------------------------------------------
// Session + message helpers (fail-closed).
// ---------------------------------------------------------------------------

export const MAX_SESSION_PARENT_DEPTH = 10

/**
 * Unwrap the `{ data?: … }` envelope shape variants a promise-domain method
 * might return. Returns `undefined` when unusable.
 */
function unwrapData(value: unknown): { data?: unknown } | undefined {
  if (value === undefined || value === null) return undefined
  if (Array.isArray(value)) return { data: value }
  if (typeof value === "object") {
    const obj = value as { data?: unknown }
    if ("data" in obj) return obj
    return { data: value }
  }
  return undefined
}

async function sessionGet(
  session: V2SessionDomain | undefined,
  sessionID: string,
): Promise<{ id?: string; parentID?: string } | null> {
  if (!session || typeof session.get !== "function") return null
  try {
    const response = await session.get({ sessionID } as never)
    const data = unwrapData(response)?.data as
      | { id?: unknown; parentID?: unknown }
      | undefined
    if (!data || typeof data !== "object") return null
    const out: { id?: string; parentID?: string } = {}
    if (typeof data.id === "string") out.id = data.id
    if (typeof data.parentID === "string" && data.parentID.length > 0) {
      out.parentID = data.parentID
    }
    return out
  } catch {
    return null
  }
}

/**
 * Walk the parentID chain up to the root session (same fail-closed contract
 * as ui/session-tree.ts: any failure returns null and the caller fails
 * closed without touching the TUI prompt).
 */
export async function resolveRootSessionID(
  session: V2SessionDomain | undefined,
  sessionID: string,
): Promise<string | null> {
  const seen = new Set<string>()
  let current = sessionID
  for (let hops = 0; hops <= MAX_SESSION_PARENT_DEPTH; hops++) {
    if (seen.has(current)) return null
    seen.add(current)
    const info = await sessionGet(session, current)
    if (info === null) return null
    if (!info.parentID) return current
    current = info.parentID
  }
  return null
}

/**
 * Read the root session's context messages and map them to the light shapes
 * the classifier prompt builder consumes: chronological user texts (V2
 * `type: "user"` carries `text` directly) and the latest assistant model ref
 * (V2 `model: { providerID, id }`).
 */
export async function loadRecentContextMessages(
  session: V2SessionDomain | undefined,
  rootSessionID: string,
  contextMessageCount: number,
): Promise<{
  userMessages: string[]
  fallbackModel: ModelRef | undefined
}> {
  if (!session || typeof session.context !== "function") {
    return { userMessages: [], fallbackModel: undefined }
  }

  let raw: unknown
  try {
    raw = await session.context({ sessionID: rootSessionID } as never)
  } catch {
    return { userMessages: [], fallbackModel: undefined }
  }

  const messages = unwrapData(raw)?.data
  if (!Array.isArray(messages)) {
    return { userMessages: [], fallbackModel: undefined }
  }

  const userMessages: string[] = []
  let fallbackModel: ModelRef | undefined = undefined
  for (const entry of messages as SessionMessageLike[]) {
    if (!entry || typeof entry !== "object") continue
    if (entry.type === "user" || entry.role === "user") {
      const text =
        typeof entry.text === "string"
          ? entry.text
          : coalesceTextParts(entry)
      const sanitized = sanitizeUserMessageForClassifier(text)
      if (sanitized.length > 0) userMessages.push(sanitized)
      continue
    }
    if (
      entry.type === "assistant" ||
      (entry.role === "assistant" && fallbackModel === undefined)
    ) {
      const model = entry.model as
        | { providerID?: unknown; id?: unknown }
        | undefined
      const providerID =
        typeof model?.providerID === "string"
          ? model.providerID
          : typeof entry.providerID === "string"
            ? entry.providerID
            : undefined
      const modelID =
        typeof model?.id === "string"
          ? model.id
          : typeof entry.modelID === "string"
            ? entry.modelID
            : undefined
      if (
        typeof providerID === "string" &&
        providerID.length > 0 &&
        typeof modelID === "string" &&
        modelID.length > 0
      ) {
        fallbackModel = { providerID, modelID }
      }
    }
  }

  return {
    userMessages: userMessages.slice(-Math.max(0, contextMessageCount)),
    fallbackModel,
  }
}

function coalesceTextParts(entry: SessionMessageLike): string {
  const parts = [...(entry.content ?? []), ...(entry.parts ?? [])]
  return parts
    .filter(
      (part): part is { type: string; text: string } =>
        typeof (part as { type?: unknown }).type === "string" &&
        (part as { type?: string }).type === "text" &&
        typeof (part as { text?: unknown }).text === "string",
    )
    .map((part) => part.text)
    .join("\n")
}

// ---------------------------------------------------------------------------
// Classification via the stateless generate.text endpoint.
// ---------------------------------------------------------------------------

const FORMAT_CORRECTION_INSTRUCTION = `

<format_correction>
Your previous response did not match the required format.
Please answer ONLY in this exact format, with no other text:
VERDICT: <SAFE|RISKY>
REASON: <one short sentence>
Classify the original subject again now.
</format_correction>`

/**
 * Fold the classifier system prompt into the single string generate.text
 * accepts (the endpoint has no `system` field).
 */
export function composeWithSystem(systemPrompt: string, userPrompt: string): string {
  return `${systemPrompt}\n\n---\n\n${userPrompt}`
}

/** Internal classification outcome. */
type ClassifyResult =
  | { ok: true; verdict: Verdict }
  | { ok: false; failureClass: "timeout" | "error" }

/**
 * Run the classifier through the stateless generate.text endpoint with the
 * same retry semantics as classifier/classify.ts: timeouts and malformed
 * responses retry (the latter with the format-correction block), hard
 * failures do not.
 */
export async function classifyViaGenerate(
  generate: GenerateDomainLike | undefined,
  args: {
    systemPrompt: string
    userPrompt: string
    model: ModelRef
    timeoutMs: number
    retries: number
  },
): Promise<ClassifyResult> {
  if (!generate || typeof generate.text !== "function") {
    return { ok: false, failureClass: "error" }
  }

  const maxAttempts = Math.max(0, args.retries) + 1
  let priorWasMalformed = false
  let lastFailure: "timeout" | "error" = "error"

  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    const userPrompt = priorWasMalformed
      ? args.userPrompt + FORMAT_CORRECTION_INSTRUCTION
      : args.userPrompt
    const prompt = composeWithSystem(args.systemPrompt, userPrompt)

    const call = generate.text({
      prompt,
      model: { providerID: args.model.providerID, id: args.model.modelID },
    } as never) as Promise<{ data?: { text?: string } } | undefined>

    const response: { data?: { text?: string } } | undefined | null =
      await raceTimeout(call, args.timeoutMs)

    if (response === null) {
      lastFailure = "timeout"
      priorWasMalformed = false
      continue // retryable timeout
    }
    if (response === undefined) {
      lastFailure = "error"
      break // hard failure: call rejected with no usable response
    }

    const text =
      typeof response.data?.text === "string" ? response.data.text : ""
    const verdict = parseVerdict(text)
    if (verdict) return { ok: true, verdict }

    lastFailure = "error"
    priorWasMalformed = true // malformed → retry with format correction
  }

  return { ok: false, failureClass: lastFailure }
}

/**
 * Race a promise against a timeout. Resolves `null` when the timeout fired,
 * `undefined` when the promise rejected, else the result.
 */
export async function raceTimeout<T>(
  promise: Promise<T>,
  timeoutMs: number,
): Promise<T | undefined | null> {
  let timer: NodeJS.Timeout | undefined
  const timeout = new Promise<null>((resolve) => {
    timer = setTimeout(() => resolve(null), timeoutMs)
  })
  try {
    return (await Promise.race([promise, timeout])) ?? null
  } catch {
    return undefined
  } finally {
    if (timer) clearTimeout(timer)
  }
}

// ---------------------------------------------------------------------------
// Post-ask notifications (risky / classifier-failure paths).
// ---------------------------------------------------------------------------

/** One pending classification outcome waiting for the ask row to exist. */
type BridgeEntry = {
  kind: "risky" | "failure"
  subject: string
  subjectLabel: "command" | "path"
  reason: string | null
  failureClass?: "timeout" | "error"
  suppressedCount?: number
  sound: boolean
}

const APPROVE_LABEL = "Approve"
const REJECT_LABEL = "Reject"
const COMMAND_DISPLAY_MAX = 180

function displaySubject(subject: string, max: number): string {
  return subject.length > max ? subject.slice(0, max) + "…" : subject
}

async function replyToPermission(
  permission: V2PermissionDomain,
  sessionID: string,
  requestID: string,
  decision: "once" | "always" | "reject",
  log: Logger,
): Promise<void> {
  try {
    await permission.reply({ sessionID, requestID, decision } as never)
    log.info("notification reply resolved", {
      sessionID,
      requestID,
      decision,
    })
  } catch (e) {
    log.error("notification reply failed (TUI prompt stays as fallback)", {
      sessionID,
      requestID,
      decision,
      error: e instanceof Error ? e.message : String(e),
    })
  }
}

// ---------------------------------------------------------------------------
// The V2 hooks: evaluate + event loop.
// ---------------------------------------------------------------------------

type V2SharedState = {
  config: DelegatedAccessConfig
  log: Logger
  session: V2SessionDomain | undefined
  generate: GenerateDomainLike | undefined
  permission: V2PermissionDomain | undefined
  directoryVerdictCache: DirectoryVerdictCache
  approvalHistory: ApprovalHistoryStore
  pendingSubjects: PendingSubjectsMap
  failureNotifyRateLimiter: FailureNotifyRateLimiter
  getRepoContext: () => Promise<DualRepoContext | RepoContext | null>
  pushBridge: (key: string, entry: BridgeEntry) => void
  bridgeKey: (sessionID: string, action: string, resources: string[]) => string
}

/** The evaluate callback — the core of the V2 plugin. */
export async function evaluatePermission(
  ev: PermissionEvaluationLike,
  deps: V2SharedState,
): Promise<void> {
  const { config, log } = deps
  const info = evaluationToInfo(ev)
  const base = {
    sessionID: info.sessionID,
    action: ev.action,
    resources: ev.resources,
  }

  // The conversation context only understands bash + external_directory
  // classification; everything else falls through. We never touch `effect`
  // for those, so the engine's own default (from the ruleset) applies.
  const isBash = BASH_ACTION_MATCHES.has(ev.action)
  const isDirectory = ev.action === EXTERNAL_DIRECTORY_ACTION
  if (!isBash && !isDirectory) {
    log.debug("evaluate: unsupported action (leaving engine default)", base)
    return
  }
  if (isDirectory && !config.externalDirectoryEnabled) {
    log.debug("evaluate: external_directory auto-approval disabled", base)
    return
  }

  if (!config.enabled) {
    log.debug("evaluate: plugin disabled", base)
    return
  }

  if (info.command === null) {
    log.info("evaluate: no classifiable subject in resources", base)
    return
  }

  // Root-session resolution — a subagent's bash permission is classified
  // against the human's root-session context, never the subagent's
  // dispatcher prompts. Fail-closed: null → leave the TUI prompt alone.
  const rootSessionID = await resolveRootSessionID(
    deps.session,
    info.sessionID,
  )
  if (rootSessionID === null) {
    log.warn("evaluate: could not resolve root session (fail-closed)", base)
    return
  }
  if (rootSessionID !== info.sessionID) {
    log.info("evaluate: resolved subagent to root session", {
      ...base,
      rootSessionID,
    })
  }

  const directoryPatterns = isDirectory ? info.patterns : undefined
  if (isDirectory && directoryPatterns) {
    const cacheKey = DirectoryVerdictCache.keyFor(directoryPatterns)
    const cached = cacheKey
      ? deps.directoryVerdictCache.get(cacheKey)
      : undefined
    if (cached) {
      log.info("evaluate: directory cache hit (skipping classifier)", {
        ...base,
        subject: displaySubject(info.command, 120),
        cachedVerdict: cached.verdict.verdict,
        cachedReason: cached.verdict.reason,
      })
      if (cached.verdict.verdict === "SAFE") {
        applyAllow(ev, cached.verdict.reason)
      }
      return
    }
  }

  const { userMessages, fallbackModel } = await loadRecentContextMessages(
    deps.session,
    rootSessionID,
    config.contextMessageCount,
  )

  // Classifier model: options override first, then the latest assistant
  // message's model ref as a fallback heurisic.
  const model = resolveClassifierModel({
    configOverride: config.classifierModel,
    sessionModel: fallbackModel,
  })
  if (!model) {
    log.warn("evaluate: no classifier model resolvable (fail-closed)", {
      ...base,
      hasConfigOverride: Boolean(config.classifierModel),
      hasFallbackModel: Boolean(fallbackModel),
    })
    return
  }

  const priorApprovals =
    config.approvalHistoryEnabled && config.approvalHistoryMax > 0
      ? deps.approvalHistory.recent(rootSessionID, config.approvalHistoryMax)
      : []

  let repoContext: DualRepoContext | RepoContext | null = null
  try {
    repoContext = await deps.getRepoContext()
  } catch (e) {
    log.warn("evaluate: repo context unavailable (continuing)", {
      ...base,
      error: e instanceof Error ? e.message : String(e),
    })
    repoContext = null
  }

  log.info("evaluate: classifying", {
    ...base,
    subject: displaySubject(info.command, 120),
    subjectLabel: info.subjectLabel,
    classifierModel: `${model.providerID}/${model.modelID}`,
    sessionBranch: pickBranch(repoContext, "pinned"),
    currentBranch: pickBranch(repoContext, "current"),
    userMessageCount: userMessages.length,
    priorApprovalCount: priorApprovals.length,
  })

  const userPrompt = isDirectory
    ? buildDirectoryClassifierUserPrompt({
        subject: info.command,
        userMessages,
        repoContext: repoContext ?? null,
        priorApprovals,
      })
    : buildClassifierUserPrompt({
        command: info.command,
        userMessages,
        repoContext: repoContext ?? null,
        priorApprovals,
      })

  const result = await classifyViaGenerate(deps.generate, {
    systemPrompt: isDirectory
      ? DIRECTORY_CLASSIFIER_SYSTEM_PROMPT
      : CLASSIFIER_SYSTEM_PROMPT,
    userPrompt,
    model,
    timeoutMs: config.classifierTimeoutMs,
    retries: config.classifierRetries,
  })

  if (!result.ok) {
    const decision = deps.failureNotifyRateLimiter.register(
      result.failureClass,
      Date.now(),
    )
    log.warn("evaluate: classifier failed (fail-closed → ask)", {
      ...base,
      failureClass: result.failureClass,
      notify: decision.notify,
      suppressedCount: decision.suppressedCount,
    })
    if (decision.notify) {
      deps.pushBridge(deps.bridgeKey(info.sessionID, ev.action, info.patterns), {
        kind: "failure",
        subject: info.command,
        subjectLabel: info.subjectLabel,
        reason: null,
        failureClass: result.failureClass,
        suppressedCount: decision.suppressedCount,
        sound: config.notificationSound,
      })
    }
    return
  }

  const verdict = result.verdict
  log.info("evaluate: classifier verdict", {
    ...base,
    verdict: verdict.verdict,
    reason: verdict.reason,
  })

  if (verdict.verdict === "SAFE") {
    if (isDirectory && directoryPatterns) {
      const cacheKey = DirectoryVerdictCache.keyFor(directoryPatterns)
      if (cacheKey) {
        deps.directoryVerdictCache.set(
          cacheKey,
          verdict,
          config.directoryVerdictCacheTtlMs,
        )
      }
    }
    applyAllow(ev, verdict.reason)
    void notifySafePath({
      config,
      log,
      subject: info.command,
      reason: verdict.reason,
    })
    return
  }

  // RISKY — leave the engine default ("ask") and bridge the ask row's ID so
  // the Approve/Reject notification can resolve it once permission.asked
  // hands it to us.
  log.info("evaluate: risky — escalating via TUI + notification", base)
  deps.pushBridge(deps.bridgeKey(info.sessionID, ev.action, info.patterns), {
    kind: "risky",
    subject: info.command,
    subjectLabel: info.subjectLabel,
    reason: verdict.reason,
    sound: config.notificationSound,
  })
}

function isExperiApprovalAvailable(
  deps: Pick<V2SharedState, "approvalHistory">,
): boolean {
  return typeof deps.approvalHistory.recent === "function"
}

function pickBranch(
  context: DualRepoContext | RepoContext | null,
  side: "pinned" | "current",
): string | null {
  if (context === null) return null
  if (isDualRepoContext(context)) {
    const branch = context[side]?.branch
    return typeof branch === "string" ? branch : null
  }
  if (side === "pinned") return context.branch
  return null
}

/** Apply the SAFE verdict: mutate effect + message on the evaluation. */
export function applyAllow(ev: PermissionEvaluationLike, reason: string): void {
  const mutable = ev as { effect?: PermissionEffect; message?: unknown }
  mutable.effect = "allow"
  mutable.message = `delegated-access: auto-approved — ${reason}`.slice(0, 300)
}

/** Fire-and-forget informational notification for an auto-approval. */
async function notifySafePath(args: {
  config: DelegatedAccessConfig
  log: Logger
  subject: string
  reason: string
}): Promise<void> {
  const { config, log, subject, reason } = args
  try {
    await sendNotification({
      title: "delegated-access: auto-approving",
      message: `${displaySubject(subject, 140)} — ${reason}`,
      sound: config.notificationSound,
    })
    log.info("safe notification posted", {
      subject: displaySubject(subject, 140),
    })
  } catch (e) {
    log.debug("safe notification failed (approval unaffected)", {
      error: e instanceof Error ? e.message : String(e),
    })
  }
}

// ---------------------------------------------------------------------------
// The V2 plugin definition.
// ---------------------------------------------------------------------------

/**
 * The V2 default definition. Same id as the package name so server-side
 * dedupe/logging stays greppable against the V1 naming.
 */
export type V2Setup = {
  id: string
  setup: (context: unknown) => Promise<(() => Promise<void>) | void>
}

/**
 * The V2 default definition. Same id as the package so server-side
 * dedupe/logs stay greppable against the V1 naming.
 *
 * Typing note: the shipped @opencode-ai/plugin 1.19.5 .d.ts narrows `setup`'s
 * result to `void | Promise<void>` — the runtime (and the server adapter)
 * treat an async-function return as a cleanup/dispose awaitable, and that is
 * what this setup returns. The narrow type is erased here rather than
 * importing broader (as-yet-unpublished) context types.
 */
export const DelegatedAccessV2 = define({
  id: "opencode-delegated-access",
  setup: setupImpl,
} as unknown as Parameters<typeof define>[0])

async function setupImpl(
  hostContext: unknown,
): Promise<(() => Promise<void>) | void> {
    const ctx = hostContext as V2PluginContext
    const log = consoleLogger()

    const worktree =
      (ctx.location && typeof ctx.location.directory === "string"
        ? ctx.location.directory
        : undefined) ?? process.cwd()
    const shell = childProcessShell()

    // Live repo-context cache (same TTL + in-flight coalescing as V1) over
    // the child-process shell shim, plus the session-pinned snapshot that
    // shares the first fetch through the cache.
    const repoContextCache = new RepoContextCache({ $: shell })
    const sessionRepoContext = new SessionRepoContext({
      worktree,
      fetcher: (cwd) => repoContextCache.get(cwd),
    })

    // Config: parse options exactly like V1 (options object → defaults on
    // invalid shapes).
    let config: DelegatedAccessConfig
    try {
      config = parseConfig(ctx.options)
      log.info("v2 plugin loaded", {
        worktree,
        enabled: config.enabled,
        classifierModel: config.classifierModel,
        contextMessageCount: config.contextMessageCount,
        classifierTimeoutMs: config.classifierTimeoutMs,
        classifierRetries: config.classifierRetries,
        externalDirectoryEnabled: config.externalDirectoryEnabled,
        approvalHistoryEnabled: config.approvalHistoryEnabled,
        safeCountdownMs: config.safeCountdownMs,
      })
    } catch (e) {
      config = parseConfig(undefined)
      log.warn("invalid plugin options; using defaults", {
        error: e instanceof Error ? e.message : String(e),
      })
    }

    // Shared state — same instance contracts the V1 factory built.
    const directoryVerdictCache = new DirectoryVerdictCache()
    const approvalHistory = new ApprovalHistoryStore({
      maxPerSession: config.approvalHistoryMax,
    })
    const pendingSubjects = new PendingSubjectsMap()
    const failureNotifyRateLimiter = new FailureNotifyRateLimiter({
      cooldownMs: config.classifierFailureNotifyCooldownMs,
    })

    const deps: V2SharedState = {
      config,
      log,
      session: ctx.session,
      generate: ctx.experimental?.generate ?? ctx.generate,
      permission: ctx.permission,
      directoryVerdictCache,
      approvalHistory,
      pendingSubjects,
      failureNotifyRateLimiter,
      getRepoContext: async () => {
        const [pinned, current] = await Promise.all([
          sessionRepoContext.getPinned(),
          repoContextCache.get(worktree),
        ])
        if (pinned === null && current === null) return null
        return { pinned, current }
      },
      pushBridge: pushBridgeEntry,
      bridgeKey: buildBridgeKey,
    }

    // Bridge: entries awaiting the permission ID the ask row gets.
    const bridges = new Map<string, BridgeEntry[]>()
    const BRIDGE_TTL_MS = 60_000
    const bridgeAges = new Map<string, number>()

    function buildBridgeKey(
      sessionID: string,
      action: string,
      resources: string[],
    ): string {
      return `${sessionID}::${action}::${resources.join("|")}`
    }

    function pushBridgeEntry(key: string, entry: BridgeEntry): void {
      const now = Date.now()
      // Sweep stale entries so long-lived sessions can't grow the map.
      for (const [k, age] of bridgeAges) {
        if (now - age > BRIDGE_TTL_MS) {
          bridgeAges.delete(k)
          bridges.delete(k)
        }
      }
      const list = bridges.get(key) ?? []
      list.push(entry)
      bridges.set(key, list)
      bridgeAges.set(key, now)
      log.debug("bridge entry pushed", { key, kind: entry.kind })
    }

    function takeBridgeEntry(key: string): BridgeEntry | undefined {
      const list = bridges.get(key)
      const entry = list?.shift()
      if (list && list.length === 0) {
        bridges.delete(key)
        bridgeAges.delete(key)
      } else if (list) {
        bridgeAges.set(key, Date.now())
      }
      return entry
    }

    // ---- Permission evaluation hook --------------------------------------
    let evaluateRegistration: RegistrationLike | undefined
    if (
      config.enabled &&
      ctx.permission &&
      typeof ctx.permission.hook === "function"
    ) {
      try {
        const registration = await ctx.permission.hook("evaluate", (ev) =>
          evaluatePermission(ev, {
            ...deps,
            bridgeKey: buildBridgeKey,
            pushBridge: pushBridgeEntry,
          }),
        )
        evaluateRegistration = registration
        log.info("permission evaluate hook registered", {})
      } catch (e) {
        log.warn(
          "permission.hook('evaluate') registration failed — plugin inert",
          { error: e instanceof Error ? e.message : String(e) },
        )
      }
    } else if (!config.enabled) {
      log.info("plugin disabled via options; evaluate hook not registered", {})
    } else {
      log.warn("ctx.permission.hook unavailable — plugin inert", {})
    }

    // ---- Event loop (asked / replied) ------------------------------------
    const controller = new AbortController()
    if (ctx.event && typeof ctx.event.subscribe === "function") {
      try {
        const events = ctx.event.subscribe({
          types: ["permission.asked", "permission.replied"],
          signal: controller.signal,
        } as never) as AsyncIterable<Record<string, unknown>>

        void (async () => {
          for await (const event of events) {
            await handleMessageEvent(event)
          }
        })().catch((e: unknown) => {
          if (controller.signal.aborted) return
          log.error("event stream ended unexpectedly", {
            error: e instanceof Error ? e.message : String(e),
          })
        })
        log.info("event subscription established", {})
      } catch (e) {
        log.warn("event subscription failed (history capture disabled)", {
          error: e instanceof Error ? e.message : String(e),
        })
      }
    } else {
      log.warn("ctx.event.subscribe unavailable; history capture disabled", {})
    }

    // Event payload wrapper: OpenCode event objects are encoded as
    // `{ id, created, metadata, type, data }` (V2) while earlier surfaces
    // used `.properties`. Probe both, then the event itself.
    function eventPayloadOf(event: Record<string, unknown>): unknown {
      if ("data" in event) return event.data
      if ("properties" in event) return event.properties
      return event
    }

    async function handleMessageEvent(
      rawEvent: Record<string, unknown>,
    ): Promise<void> {
      try {
        const type = rawEvent?.type
        if (type === "permission.asked") {
          const data = eventPayloadOf(rawEvent) as
            | {
                sessionID?: unknown
                action?: unknown
                resources?: unknown
                id?: unknown
              }
            | undefined
          if (
            !data ||
            typeof data !== "object" ||
            typeof data.sessionID !== "string" ||
            typeof data.action !== "string" ||
            typeof data.id !== "string"
          ) {
            log.debug("permission.asked: unusable payload", {
              raw: data as unknown,
            })
            return
          }
          const resources = Array.isArray(data.resources)
            ? (data.resources as unknown[]).filter(
                (r): r is string => typeof r === "string",
              )
            : []
          // Extract into typed consts BEFORE the fire-and-forget closures so
          // TS's property-narrowing limitations can't bite inside them.
          type AskRow = {
            sessionID: string
            action: string
            id: string
            resources: string[]
          }
          const ask: AskRow = {
            sessionID: data.sessionID,
            action: data.action,
            id: data.id,
            resources,
          }
          const key = buildBridgeKey(ask.sessionID, ask.action, ask.resources)
          const entry = takeBridgeEntry(key)
          if (!entry) {
            log.debug("permission.asked: no pending bridge entry", {
              sessionID: ask.sessionID,
              action: ask.action,
              permissionID: ask.id,
            })
            return
          }
          log.info("consumed bridge entry for ask row", {
            permissionID: ask.id,
            kind: entry.kind,
            subject: displaySubject(entry.subject, 120),
          })
          pendingSubjects.set(ask.id, {
            rootSessionID: ask.sessionID,
            subject: entry.subject,
            subjectLabel: entry.subjectLabel,
            classifierVerdict: entry.kind === "risky" ? "RISKY" : null,
            classifierReason: entry.reason,
            autoApproved: false,
          })
          if (entry.kind === "risky") {
            void (async () => {
              try {
                const result = await sendNotification({
                  title: "delegated-access: review " + entry.subjectLabel,
                  message: `${displaySubject(entry.subject, COMMAND_DISPLAY_MAX)} (${entry.reason})`,
                  actions: [APPROVE_LABEL, REJECT_LABEL],
                  sound: entry.sound,
                  timeoutSec: 60,
                })
                if (result.type !== "action") return
                const decision =
                  result.label === APPROVE_LABEL
                    ? "once"
                    : result.label === REJECT_LABEL
                      ? "reject"
                      : undefined
                if (
                  decision === undefined ||
                  typeof ctx.permission?.reply !== "function"
                ) {
                  return
                }
                await replyToPermission(
                  ctx.permission,
                  ask.sessionID,
                  ask.id,
                  decision,
                  log,
                )
              } catch (e) {
                log.warn("risky notification flow failed (TUI stays)", {
                  permissionID: ask.id,
                  error: e instanceof Error ? e.message : String(e),
                })
              }
            })()
          } else {
            // Classifier-failure notification: Reject-only, one-shot.
            void (async () => {
              try {
                const classLabel =
                  entry.failureClass === "timeout"
                    ? "classifier timed out"
                    : "classifier error"
                const burstSuffix =
                  (entry.suppressedCount ?? 0) > 0
                    ? ` (${(entry.suppressedCount ?? 0) + 1} recent failures)`
                    : ""
                const result = await sendNotification({
                  title: `delegated-access: ${classLabel} — review in TUI`,
                  message: `${displaySubject(entry.subject, 160)}${burstSuffix}`,
                  actions: [REJECT_LABEL],
                  sound: entry.sound,
                  timeoutSec: 60,
                })
                if (
                  result.type === "action" &&
                  result.label === REJECT_LABEL &&
                  typeof ctx.permission?.reply === "function"
                ) {
                  await replyToPermission(
                    ctx.permission,
                    ask.sessionID,
                    ask.id,
                    "reject",
                    log,
                  )
                }
              } catch (e) {
                log.warn("failure notification flow failed (TUI stays)", {
                  permissionID: ask.id,
                  error: e instanceof Error ? e.message : String(e),
                })
              }
            })()
          }
          return
        }

        if (type === "permission.replied") {
          const canonical = normalizeV2RepliedProperties(
            eventPayloadOf(rawEvent),
          )
          if (canonical === null) {
            log.warn("permission.replied: malformed event shape", {
              rawEvent: Object.keys(rawEvent ?? {}),
            })
            return
          }
          handlePermissionReplied(canonical, {
            pendingSubjects,
            approvalHistory,
            config,
            log: v1LoggerAdapter(log),
          })
          return
        }
      } catch (e) {
        log.error("event handling threw", {
          error: e instanceof Error ? e.message : String(e),
        })
      }
    }

    log.info("v2 setup complete", {
      evaluateHook: Boolean(evaluateRegistration),
      eventStream: Boolean(ctx.event),
    })

    // ---- Cleanup ----------------------------------------------------------
    return async () => {
      controller.abort()
      if (evaluateRegistration) {
        try {
          await evaluateRegistration.dispose()
        } catch {
          // Dispose is best-effort.
        }
      }
      log.info("v2 plugin unloaded; event stream aborted", {})
    }
  }

/**
 * The V2 Logger and the V1 Logger are structurally identical (same method
 * signatures) — adapt only the branding so V1 handler lines match V1 greps.
 */function v1LoggerAdapter(log: Logger): import("./log.ts").Logger {
  return {
    debug: (m, e) => log.debug(m, e),
    info: (m, e) => log.info(m, e),
    warn: (m, e) => log.warn(m, e),
    error: (m, e) => log.error(m, e),
  }
}

export default DelegatedAccessV2
