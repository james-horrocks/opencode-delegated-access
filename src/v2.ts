/**
 * OpenCode V2 entrypoint for the delegated-access plugin.
 *
 * # Why this file exists
 *
 * OpenCode 2.x's plugin loader rejects the V1 file's function default export
 * (`Plugin must export a default definition with an id and an effect or setup
 * function`), so upstream has been INERT on the fleet's OpenCode 2 hosts:
 * fetched + cached at startup, then skipped at registration. This module
 * supplies the V2 entrypoint the 2.x loader wants, and is vendored by the
 * nix-configs as ~/.config/opencode/plugins/opencode-delegated-access (the
 * loader resolves a package's `<name>/server` exports subpath first, so the
 * 2.x host loads THIS entry while "." keeps the V1 default for 1.x hosts).
 *
 * # Zero runtime dependencies — why this file re-implements small helpers
 *
 * A vendored plugins/ directory lives under a path with NO ancestor
 * node_modules chain (for the nix deployment that path is a /nix/store
 * source dir). The loader resolves imports relative to the importing file,
 * so any external VALUE import here dies with "Cannot find package".
 * Everything below is therefore dependency-free: upstream's define() is a
 * pure identity helper (packages/plugin/src/promise/plugin.ts) so we ship a
 * local typed copy; config parsing mirrors config.ts's DEFAULT_CONFIG with
 * per-field fallbacks; the bash/gh shell runner is a child_process shim
 * standing in for the Bun `$` the V2 context lacks; and notifications go
 * through Linux `notify-send` (node-notifier's rich-action behaviour is not
 * portable through child_process here). Type-only imports (sdk/… shape
 * declarations in local files) are erased at load and safe.
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
 *   - Reply/history events    → `ctx.event.subscribe({ types:
 *     ["permission.asked", "permission.replied"] })`; V2 replied shapes are
 *     normalised to the canonical { sessionID, permissionID, response } the
 *     shared approval-history handler consumes (re-implemented here locally
 *     to keep the dependency-free contract).
 *   - Repo context (git/gh)   → `childProcessShell()` shim in v2-shell.ts.
 *   - Logging                 → console with the "[delegated-access]"
 *     prefix (fire-and-forget; the V2 promise context has no
 *     `client.app.log`).
 *
 * # Deliberate behaviour deltas vs V1 (documented — do not "fix" silently)
 *
 *   1. SAFE verdicts set `effect = "allow"` during evaluation: auto-approval
 *      is pre-prompt, so there is no live TUI prompt to cancel during the
 *      old `safeCountdownMs` window. All notifications are informational
 *      (no Approve/Reject buttons — interactive notification action
 *      callbacks are not portable here); RISKY and classifier-failure
 *      subjects fall through to the engine's own "ask" and the human still
 *      resolves them in the TUI.
 *   2. The session-model auto-detection uses the latest assistant message
 *      from the root session's context; `options.classifierModel`
 *      (delivered for tuple-config installs) remains authoritative.
 *   3. Approval-recorded history: replies match the ask row via
 *      permission.asked's permission id bridged from this evaluation
 *      outcome (the evaluate input itself carries no row id).
 */

import type { DelegatedAccessConfig } from "./config.ts"
import {
  CLASSIFIER_SYSTEM_PROMPT,
  buildClassifierUserPrompt,
  DIRECTORY_CLASSIFIER_SYSTEM_PROMPT,
  buildDirectoryClassifierUserPrompt,
} from "./classifier/prompt.ts"
import { parseVerdict, type Verdict } from "./classifier/parse.ts"
import { resolveClassifierModel, type ModelRef } from "./classifier/model.ts"
import { sanitizeUserMessageForClassifier } from "./ui/messages.ts"
import { ApprovalHistoryStore } from "./permission/approval-history.ts"
import { DirectoryVerdictCache } from "./permission/directory-cache.ts"
import { PendingSubjectsMap } from "./permission/pending-subjects.ts"
import {
  isDualRepoContext,
  RepoContextCache,
  type DualRepoContext,
  type RepoContext,
} from "./repo-context.ts"
import { SessionRepoContext } from "./session-repo-context.ts"
import { childProcessShell } from "./v2-shell.ts"

// ---------------------------------------------------------------------------
// Local `define` helper (promise-API compatible).
//
// Upstream's define is literally `function define(plugin) { return
// plugin; }` — an identity helper whose only purpose is a typed object
// literal. Keep the structural typing in sync if the loader's server-side
// schema for the default definition ever grows new required fields.
// ---------------------------------------------------------------------------

/**
 * The server-side validation the OpenCode 2.x loader performs on the default
 * export: an object with a string `id` and a `setup` (or `effect`) function.
 */
export type PromisePluginDefinition = {
  id: string
  setup: (
    context: unknown,
  ) => Promise<(() => Promise<void>) | void> | (() => Promise<void>) | void
}

export function define<T extends PromisePluginDefinition>(plugin: T): T {
  return plugin
}

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
 * the approval-history store consumes. V2 field names (requestID/reply) map
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
// Config defaults (zod-free — see the file header).
// ---------------------------------------------------------------------------


const V2_DEFAULT_NUMBERS = {
  contextMessageCount: 3,
  safeCountdownMs: 5_000,
  classifierTimeoutMs: 15_000,
  classifierRetries: 1,
  directoryVerdictCacheTtlMs: 60_000,
  approvalHistoryMax: 20,
  classifierFailureNotifyCooldownMs: 60_000,
}

const V2_DEFAULT_BOOLEANS = {
  enabled: true,
  notificationSound: true,
  externalDirectoryEnabled: true,
  approvalHistoryEnabled: true,
  notifyOnClassifierFailure: true,
}

/**
 * Permissive numbers with clamping to the V1 zod ranges; unknown values
 * fall back to the default. Mirrors config.ts's DEFAULT_CONFIG exactly.
 */
export function parseConfigV2(input: unknown): DelegatedAccessConfig {
  const o =
    input && typeof input === "object"
      ? (input as Record<string, unknown>)
      : {}
  const num = (
    key: keyof typeof V2_DEFAULT_NUMBERS,
    min: number,
    max: number,
  ): number => {
    const v = o[key]
    return typeof v === "number" && Number.isFinite(v)
      ? Math.min(max, Math.max(min, Math.round(v)))
      : V2_DEFAULT_NUMBERS[key]
  }
  const bool = (key: keyof typeof V2_DEFAULT_BOOLEANS): boolean => {
    const v = o[key]
    return typeof v === "boolean" ? v : V2_DEFAULT_BOOLEANS[key]
  }
  return {
    enabled: bool("enabled"),
    contextMessageCount: num("contextMessageCount", 0, 20),
    safeCountdownMs: num("safeCountdownMs", 0, 60_000),
    classifierModel:
      typeof o.classifierModel === "string" &&
      o.classifierModel.trim().length > 0
        ? o.classifierModel.trim()
        : undefined,
    classifierTimeoutMs: num("classifierTimeoutMs", 500, 60_000),
    classifierRetries: num("classifierRetries", 0, 10),
    notificationSound: bool("notificationSound"),
    externalDirectoryEnabled: bool("externalDirectoryEnabled"),
    directoryVerdictCacheTtlMs: num("directoryVerdictCacheTtlMs", 0, 300_000),
    approvalHistoryEnabled: bool("approvalHistoryEnabled"),
    approvalHistoryMax: num("approvalHistoryMax", 0, 1000),
    notifyOnClassifierFailure: bool("notifyOnClassifierFailure"),
    classifierFailureNotifyCooldownMs: num(
      "classifierFailureNotifyCooldownMs",
      0,
      3_600_000,
    ),
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
export function composeWithSystem(
  systemPrompt: string,
  userPrompt: string,
): string {
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
// Data-only notifier (dependency-free): the Linux `notify-send` CLI, best
// effort. Action buttons are deliberately NOT used (interactive notification
// action callbacks are not portable here — see the file header); everything
// resolves via the OpenCode TUI as usual.
// ---------------------------------------------------------------------------

const SUBJECT_DISPLAY_MAX = 140

function displaySubject(subject: string, max: number): string {
  return subject.length > max ? subject.slice(0, max) + "…" : subject
}

type NotifyArgs = {
  title: string
  message: string
  /** Seconds the notification stays on screen (0 = server default). */
  timeoutSec?: number
}

/** Best-effort `notify-send`. Never throws; returns void. */
async function notifySend(args: NotifyArgs): Promise<void> {
  const { title, message, timeoutSec } = args
  try {
    const { spawn } = await import("node:child_process")
    const argv = [
      title,
      message,
      ...(timeoutSec && timeoutSec > 0
        ? ["-t", String(Math.min(2147483647, timeoutSec * 1000))]
        : []),
    ]
    await new Promise<void>((resolve) => {
      const child = spawn("notify-send", argv, {
        stdio: ["ignore", "ignore", "ignore"],
      })
      child.on("error", () => resolve())
      child.on("close", () => resolve())
    })
  } catch {
    // Notification failures must never affect the permission flow.
  }
}

// ---------------------------------------------------------------------------
// Rate limiter for classifier-failure notifications (mirror of
// permission/failure-notify.ts's FailureNotifyRateLimiter — locals kept
// dependency-free).
// ---------------------------------------------------------------------------

export type RateLimitDecision = {
  notify: boolean
  suppressedCount: number
}

/**
 * At most ONE failure notification per cooldown window; a burst of failures
 * during a sustained outage collapses into a single notification.
 */
export class V2FailureRateLimiter {
  private readonly cooldownMs: number
  private lastNotifyAt: number | null = null
  private suppressedSinceLast = 0

  constructor(opts: { cooldownMs: number }) {
    this.cooldownMs = Math.max(0, opts.cooldownMs)
  }

  register(_failureClass: string, now: number): RateLimitDecision {
    if (this.cooldownMs === 0) {
      return { notify: true, suppressedCount: 0 }
    }
    if (
      this.lastNotifyAt === null ||
      now - this.lastNotifyAt >= this.cooldownMs
    ) {
      const suppressed = this.suppressedSinceLast
      this.lastNotifyAt = now
      this.suppressedSinceLast = 0
      return { notify: true, suppressedCount: suppressed }
    }
    this.suppressedSinceLast++
    return { notify: false, suppressedCount: 0 }
  }
}

// ---------------------------------------------------------------------------
// Approval-history capture (mirror of index.ts's handlePermissionReplied —
// local copy keeps the dependency-free contract; the V1 function is
// unchanged for the V1 path).
// ---------------------------------------------------------------------------

/**
 * Pure handler for `permission.replied` events over the canonical shape.
 * Looks up the matching pending subject, filters OUR auto-approvals out,
 * and appends a human-decision entry to the approval history.
 */
export function handleRepliedForHistory(
  canonical: { sessionID: string; permissionID: string; response: string },
  deps: {
    pendingSubjects: PendingSubjectsMap
    approvalHistory: ApprovalHistoryStore
    config: DelegatedAccessConfig
    log: Logger
  },
): void {
  const { pendingSubjects, approvalHistory, config, log } = deps
  const now = Date.now()

  if (!config.approvalHistoryEnabled) {
    log.debug("permission.replied: history disabled", {
      permissionID: canonical.permissionID,
    })
    return
  }

  const pending = pendingSubjects.take(canonical.permissionID)
  if (!pending) {
    log.debug("permission.replied: no pending subject for permissionID", {
      permissionID: canonical.permissionID,
    })
    return
  }

  if (pending.autoApproved) {
    log.debug("permission.replied: skipping our own auto-approval", {
      permissionID: canonical.permissionID,
      subject: pending.subject,
    })
    return
  }

  const response = canonical.response
  if (response !== "once" && response !== "always" && response !== "reject") {
    log.warn("permission.replied: unrecognised response value", {
      permissionID: canonical.permissionID,
      response,
    })
    return
  }

  if (pending.classifierVerdict === null) {
    log.info("permission.replied: human resolved before classifier", {
      permissionID: canonical.permissionID,
      response,
    })
  }

  approvalHistory.record(pending.rootSessionID, {
    subject: pending.subject,
    subjectLabel: pending.subjectLabel,
    response,
    classifierVerdict: pending.classifierVerdict ?? "RISKY",
    classifierReason:
      pending.classifierReason ??
      "(classifier did not complete before human resolved)",
    timestamp: now,
  })

  log.info("recorded human approval decision", {
    rootSessionID: pending.rootSessionID,
    permissionID: canonical.permissionID,
    response,
    subject: pending.subject,
  })
}

// ---------------------------------------------------------------------------
// The V2 hooks: evaluate + event loop.
// ---------------------------------------------------------------------------

type V2SharedState = {
  config: DelegatedAccessConfig
  log: Logger
  session: V2SessionDomain | undefined
  generate: GenerateDomainLike | undefined
  directoryVerdictCache: DirectoryVerdictCache
  approvalHistory: ApprovalHistoryStore
  pendingSubjects: PendingSubjectsMap
  failureNotifyRateLimiter: V2FailureRateLimiter
  getRepoContext: () => Promise<DualRepoContext | RepoContext | null>
  /** key → queue of classify outcomes, consumed by permission.asked. */
  pushRowBridge: (key: string, entry: RowBridgeEntry) => void
  takeRowBridge: (key: string) => RowBridgeEntry | undefined
  bridgeKey: (sessionID: string, action: string, resources: string[]) => string
}

/** A resolved classify outcome waiting for the ask row's permission ID. */
type RowBridgeEntry = {
  subject: string
  subjectLabel: "command" | "path"
  classifierVerdict: "SAFE" | "RISKY" | null
  classifierReason: string | null
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

  // Only bash + external_directory are classified; other actions fall
  // through untouched so the engine's own ruleset default applies.
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
    if (config.notifyOnClassifierFailure && decision.notify) {
      void notifySend({
        title: `delegated-access: classifier ${
          result.failureClass === "timeout" ? "timed out" : "error"
        } — review in TUI`,
        message: `${displaySubject(info.command, 160)}${
          decision.suppressedCount > 0
            ? ` (${decision.suppressedCount + 1} recent failures)`
            : ""
        }`,
        timeoutSec: 60,
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
    if (config.notificationSound) {
      void notifySend({
        title: "delegated-access: auto-approving",
        message: `${displaySubject(info.command, SUBJECT_DISPLAY_MAX)} — ${verdict.reason}`,
      })
    }
    return
  }

  // RISKY — leave the engine default ("ask"), inform the human, and bridge
  // the ask row's permission ID so the reply event can record the decision.
  log.info("evaluate: risky — escalating via TUI + notification", base)
  if (config.notificationSound) {
    void notifySend({
      title: "delegated-access: risky subject — review in TUI",
      message: `${displaySubject(info.command, SUBJECT_DISPLAY_MAX)} — ${verdict.reason}`,
      timeoutSec: 60,
    })
  }
  deps.pushRowBridge(deps.bridgeKey(info.sessionID, ev.action, info.patterns), {
    subject: info.command,
    subjectLabel: info.subjectLabel,
    classifierVerdict: "RISKY",
    classifierReason: verdict.reason,
  })
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

// ---------------------------------------------------------------------------
// The V2 plugin definition.
// ---------------------------------------------------------------------------

/**
 * The V2 default definition. Same id as the package so server-side
 * dedupe/logging stays greppable against the V1 naming.
 *
 * Typing note: the shipped @opencode-ai/plugin .d.ts narrows `setup`'s
 * result to `void | Promise<void>` — the server runtime (and the promise
 * adapter) treats an async-function return as a cleanup/dispose awaitable,
 * which is what this setup returns; the narrow type is erased rather than
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

  // Config: parse options (tuple-config installs deliver them here;
  // vendored-dir installs deliver nothing → defaults).
  let config: DelegatedAccessConfig
  try {
    config = parseConfigV2(ctx.options)
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
    config = parseConfigV2(undefined)
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
  const failureNotifyRateLimiter = new V2FailureRateLimiter({
    cooldownMs: config.classifierFailureNotifyCooldownMs,
  })

  // Bridge: rows the classifier adjudicated, waiting for the ask row's
  // permission ID from `permission.asked` (so replies can record history).
  const rowBridges = new Map<string, RowBridgeEntry[]>()
  const ROW_BRIDGE_TTL_MS = 60_000
  const bridgeAges = new Map<string, number>()

  function buildBridgeKey(
    sessionID: string,
    action: string,
    resources: string[],
  ): string {
    return `${sessionID}::${action}::${resources.join("|")}`
  }

  function pushRowBridge(key: string, entry: RowBridgeEntry): void {
    const now = Date.now()
    for (const [k, age] of bridgeAges) {
      if (now - age > ROW_BRIDGE_TTL_MS) {
        bridgeAges.delete(k)
        rowBridges.delete(k)
      }
    }
    const list = rowBridges.get(key) ?? []
    list.push(entry)
    rowBridges.set(key, list)
    bridgeAges.set(key, now)
    log.debug("row bridge pushed", { key })
  }

  function takeRowBridge(key: string): RowBridgeEntry | undefined {
    const list = rowBridges.get(key)
    const entry = list?.shift()
    if (list && list.length === 0) {
      rowBridges.delete(key)
      bridgeAges.delete(key)
    } else if (list) {
      bridgeAges.set(key, Date.now())
    }
    return entry
  }

  const deps: V2SharedState = {
    config,
    log,
    session: ctx.session,
    generate: ctx.experimental?.generate ?? ctx.generate,
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
    pushRowBridge,
    takeRowBridge,
    bridgeKey: buildBridgeKey,
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
          pushRowBridge,
          takeRowBridge,
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

  // ---- Event loop (asked / replied) — history capture only --------------
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

  /**
   * Event payload wrapper: OpenCode event objects are encoded as
   * `{ id, created, metadata, type, data }` (V2) while earlier surfaces
   * used `.properties`. Probe both, then the event itself.
   */
  function eventPayloadOf(event: Record<string, unknown>): unknown {
    if ("data" in event) return event.data
    if ("properties" in event) return event.properties
    return event
  }

  async function handleMessageEvent(
    rawEvent: Record<string, unknown>,
  ): Promise<void> {
    try {
      if (!rawEvent || typeof rawEvent !== "object") return
      const type = rawEvent.type
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
        const ask = {
          sessionID: data.sessionID,
          action: data.action,
          id: data.id,
          resources,
        }
        const key = buildBridgeKey(ask.sessionID, ask.action, ask.resources)
        const entry = takeRowBridge(key)
        if (!entry) {
          log.debug("permission.asked: no pending row bridge entry", {
            sessionID: ask.sessionID,
            action: ask.action,
            permissionID: ask.id,
          })
          return
        }
        pendingSubjects.set(ask.id, {
          rootSessionID: ask.sessionID,
          subject: entry.subject,
          subjectLabel: entry.subjectLabel,
          classifierVerdict: entry.classifierVerdict,
          classifierReason: entry.classifierReason,
          autoApproved: false,
        })
        log.info("consumed row bridge entry for ask row", {
          permissionID: ask.id,
          subject: displaySubject(entry.subject, 120),
          verdict: entry.classifierVerdict,
        })
        return
      }

      if (type === "permission.replied") {
        const canonical = normalizeV2RepliedProperties(
          eventPayloadOf(rawEvent),
        )
        if (canonical === null) {
          log.warn("permission.replied: malformed event shape", {
            keys: Object.keys(rawEvent ?? {}),
          })
          return
        }
        handleRepliedForHistory(canonical, {
          pendingSubjects,
          approvalHistory,
          config,
          log,
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
 * The V2 Logger and the V1 Logger are structurally identical — adapt only
 * the module branding so shared V1 helper log lines still grep as V1.
 */
export default DelegatedAccessV2
