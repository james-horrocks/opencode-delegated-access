/**
 * OpenCode V2 *server* subpath entry.
 *
 * The V2 plugin host resolves a package's entrypoints as `[<name>/server,
 * <name>]` (packages/plugin/src/host.ts): a package that declares a
 * `./server` export loads THAT FILE for server plugins and leaves "." for
 * anything else. Upstream keeps "." on the V1 entry (OpenCode 1.x hosts and
 * the V1 test suite), while this subpath hands OpenCode 2.x the promise-API
 * definition it requires (define({ id, setup })).
 *
 * Files loaded through this subpath must default-export the V2 definition;
 * src/v2.ts owns that implementation and its docs.
 */
export { DelegatedAccessV2 as default } from "./v2.ts"
