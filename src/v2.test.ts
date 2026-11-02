import { describe, expect, it } from "vitest"

import {
  applyAllow,
  composeWithSystem,
  evaluationToInfo,
  normalizeV2RepliedProperties,
  raceTimeout,
} from "./v2.ts"

const sleep = (ms: number) =>
  new Promise((resolve) => setTimeout(resolve, ms))

describe("normalizeV2RepliedProperties", () => {
  it("maps V2 field names (sessionID/requestID/reply)", () => {
    expect(
      normalizeV2RepliedProperties({
        sessionID: "ses-1",
        requestID: "per-2",
        reply: "once",
      }),
    ).toEqual({
      sessionID: "ses-1",
      permissionID: "per-2",
      response: "once",
    })
  })

  it("passes through V1 field names", () => {
    expect(
      normalizeV2RepliedProperties({
        sessionID: "ses-1",
        permissionID: "per-3",
        response: "reject",
      }),
    ).toEqual({ sessionID: "ses-1", permissionID: "per-3", response: "reject" })
  })

  it("returns null for malformed payloads", () => {
    expect(normalizeV2RepliedProperties(null)).toBeNull()
    expect(normalizeV2RepliedProperties({ requestID: "per-2" })).toBeNull()
    expect(
      normalizeV2RepliedProperties({ sessionID: "ses-1", reply: "once" }),
    ).toBeNull()
    expect(
      normalizeV2RepliedProperties({ sessionID: "ses-1", requestID: null, reply: 7 }),
    ).toBeNull()
  })
})

describe("evaluationToInfo", () => {
  it("re-joins multi-segment bash commands with &&", () => {
    const info = evaluationToInfo({
      sessionID: "ses-1",
      action: "bash",
      resources: ["git add .", "git commit -m x"],
      effect: "ask",
    })
    expect(info.command).toBe("git add . && git commit -m x")
    expect(info.subjectLabel).toBe("command")
  })

  it("hands back single-segment commands unchanged", () => {
    const info = evaluationToInfo({
      sessionID: "ses-1",
      action: "bash",
      resources: ["git status"],
      effect: "ask",
    })
    expect(info.command).toBe("git status")
    expect(info.subjectLabel).toBe("command")
  })

  it("labels external_directory as a path subject", () => {
    const info = evaluationToInfo({
      sessionID: "ses-1",
      action: "external_directory",
      resources: ["/tmp/one", "/tmp/two"],
      effect: "ask",
    })
    expect(info.subjectLabel).toBe("path")
    expect(info.command).toBe("/tmp/one")
    expect(info.patterns).toEqual(["/tmp/one", "/tmp/two"])
  })

  it("ignores non-string resources", () => {
    const info = evaluationToInfo({
      sessionID: "ses-1",
      action: "bash",
      resources: [1, 2, "git status"] as unknown as string[],
      effect: "ask",
    })
    expect(info.command).toBeNull()
  })
})

describe("applyAllow", () => {
  it("mutates the evaluation to allow with a reason message", () => {
    const ev = { sessionID: "s", action: "bash", resources: [], effect: "ask" as const }
    applyAllow(ev, "read-only")
    expect(ev.effect).toBe("allow")
    expect((ev as { message?: string }).message).toContain(
      "delegated-access: auto-approved — read-only",
    )
  })
})

describe("composeWithSystem", () => {
  it("concatenates system and user prompts with a delimiter", () => {
    const merged = composeWithSystem("SYSTEM-ROLE", "USER-TASK")
    expect(merged.startsWith("SYSTEM-ROLE")).toBe(true)
    expect(merged.endsWith("USER-TASK")).toBe(true)
    expect(merged).toContain("SYSTEM-ROLE\n\n---\n\nUSER-TASK")
  })
})

describe("raceTimeout", () => {
  it("returns null when the timeout fires first", async () => {
    const result = await raceTimeout(sleep(200), 10)
    expect(result).toBeNull()
  })

  it("returns the value when the promise wins", async () => {
    const result = await raceTimeout(sleep(10).then(() => "value"), 200)
    expect(result).toBe("value")
  })

  it("returns undefined when the promise rejects", async () => {
    const result = await raceTimeout(sleep(10).then(() => {
      throw new Error("boom")
    }), 200)
    expect(result).toBeUndefined()
  })
})
