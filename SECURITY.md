# Dependency security

Run the dependency audit with the command in the README. CI uses the same command.

ChromaDB 0.6.3 has three unpatched advisories:

- [GHSA-36p7-vc44-83pf](https://github.com/advisories/GHSA-36p7-vc44-83pf): code injection through the collection-update server API.
- [GHSA-xph7-9rjv-w5fr](https://github.com/advisories/GHSA-xph7-9rjv-w5fr): missing tenant, database, and collection checks in server RBAC.
- [GHSA-2wm9-hf6c-p5cr](https://github.com/advisories/GHSA-2wm9-hf6c-p5cr): missing tenant authorization in the server API.

Docstra uses an embedded `chromadb.PersistentClient` in
`docstra/core/ingestion/storage.py`. It does not start a ChromaDB server or use
its HTTP client, authentication, or tenant authorization. These server attack
paths are not exposed by Docstra's storage configuration. Do not run a ChromaDB
server with this dependency version.

The audit excludes these advisories only while no fix is available, using
`--ignore-until-fixed`. GitHub alerts remain open. Reassess these exceptions
before adding remote ChromaDB support. Reviewed on November 13, 2026.
