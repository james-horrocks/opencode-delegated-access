/**
 * Cross-runtime shell runner shaped like Bun's `$` tag helper.
 *
 * The V1 entrypoint receives a real Bun `$` from the plugin host and passes
 * it into `RepoContextCache` for its `git` / `gh` shell-outs. The V2 promise
 * plugin context has no `$` — and the plugin may also be evaluated by a
 * Node-flavoured runtime (`packages/plugin/src/source.node.ts`) where
 * `import "bun"` does not exist. So the V2 entry exists only as a
 * `node:child_process` based runner that satisfies the exact call pattern
 * `repo-context.ts` uses:
 *
 *     $`${exe} ${rest}`.cwd(cwd).quiet().nothrow()
 *
 * Only the following Bun-tag behaviours are emulated (that is the full
 * surface `fetchRepoContext` / `RepoContextCache` rely on):
 *
 *   - Template literals contain no shell metacharacters of their own, so the
 *     argv comes exclusively from the interpolated expressions: a string
 *     expression is one argv slot; a string-array expression is One slice per
 *     element — the same no-quoting arg-packing Bun applies to array
 *     interpolation. The literals themselves (`""` and `" "`) are separators,
 *     never part of the command.
 *   - `.cwd(cwd)` (chainable, applies before execution).
 *   - `.quiet()` / `.nothrow()` chainable no-ops for our runner (stderr is
 *     already discarded, and exit codes are surfaced rather than thrown).
 *   - The settle value exposes `exitCode: number` and `text(encoding?)` for
 *     captured stdout.
 */

import { spawn } from "node:child_process"

export type ShellOutput = {
  exitCode: number
  text(encoding?: string): string
}

export type ShellPromise = Promise<ShellOutput> & {
  cwd(newCwd: string): ShellPromise
  quiet(): ShellPromise
  nothrow(): ShellPromise
}

export type BunShellLike = (
  strings: TemplateStringsArray,
  ...exprs: unknown[]
) => ShellPromise

/** Collect a single argv from the tag's expressions (Bun array packing). */
function collectArgv(exprs: unknown[]): string[] {
  const argv: string[] = []
  for (const expr of exprs) {
    if (Array.isArray(expr)) {
      for (const item of expr) {
        if (typeof item === "string") argv.push(item)
      }
    } else if (typeof expr === "string") {
      argv.push(expr)
    } else if (expr !== null && expr !== undefined) {
      argv.push(String(expr))
    }
  }
  return argv
}

/**
 * Build a shell tag from `node:child_process`. Execution starts LAZILY on the
 * first `.then`/`await` so the chainable `.cwd(...)` calls (made synchronously
 * right after tagging, before the promise settles) take effect.
 */
export function childProcessShell(): BunShellLike {
  return (_strings, ...exprs) => {
    const argv = collectArgv(exprs)
    const exe = argv[0]
    const rest = argv.slice(1)
    let cwd: string | undefined
    let started: Promise<ShellOutput> | null = null

    const run = (): Promise<ShellOutput> => {
      if (started) return started
      started = new Promise<ShellOutput>((resolve) => {
        if (typeof exe !== "string" || exe.length === 0) {
          resolve({ exitCode: 1, text: () => "" })
          return
        }
        try {
          const child = spawn(exe, rest, {
            cwd: cwd ?? process.cwd(),
            stdio: ["ignore", "pipe", "pipe"],
          })
          const stdout = child.stdout as NodeJS.ReadableStream
          let text = ""
          stdout.setEncoding("utf8")
          stdout.on("data", (chunk: unknown) => {
            text += typeof chunk === "string" ? chunk : ""
          })
          child.on("error", () => {
            // ENOENT etc. — mirrors Bun's throw-on-spawn-failure, surfaced as
            // exitCode 1 with empty text (repo-context's runQuiet handles it).
            resolve({ exitCode: 1, text: () => text })
          })
          child.on("close", (code?: number | null) => {
            resolve({ exitCode: code ?? 1, text: () => text })
          })
        } catch {
          resolve({ exitCode: 1, text: () => "" })
        }
      })
      return started
    }

    const shell = {
      then: <A, B>(
        onFulfilled?: (value: ShellOutput) => A | PromiseLike<A>,
        onRejected?: (reason: unknown) => B | PromiseLike<B>,
      ) => run().then(onFulfilled, onRejected),
      catch: <B>(onRejected?: (reason: unknown) => B | PromiseLike<B>) =>
        run().catch(onRejected),
      finally: (onFinally?: () => unknown) => run().finally(onFinally),
      cwd: (newCwd: string) => {
        cwd = newCwd
        return shell as unknown as ShellPromise
      },
      quiet: () => shell as unknown as ShellPromise,
      nothrow: () => shell as unknown as ShellPromise,
      [Symbol.toStringTag]: "Promise",
    }
    return shell as unknown as ShellPromise
  }
}
