# Dependency security

Run the dependency audit with the command in the README. CI uses the same command.

ChromaDB 1.5.9, the latest release, has four unpatched advisories:

- [GHSA-36p7-vc44-83pf](https://github.com/advisories/GHSA-36p7-vc44-83pf): code injection through the collection-update server API.
- [GHSA-xph7-9rjv-w5fr](https://github.com/advisories/GHSA-xph7-9rjv-w5fr): missing tenant, database, and collection checks in server RBAC.
- [GHSA-2wm9-hf6c-p5cr](https://github.com/advisories/GHSA-2wm9-hf6c-p5cr): missing tenant authorization in the server API.
- [GHSA-f4j7-r4q5-qw2c](https://github.com/advisories/GHSA-f4j7-r4q5-qw2c): pre-authentication code injection through the server collections endpoint.

Docstra uses an embedded `chromadb.PersistentClient` in
`docstra/core/ingestion/storage.py`. It does not start a ChromaDB server or use
its HTTP client, authentication, or tenant authorization. These server attack
paths are not exposed by Docstra's storage configuration. Do not run a ChromaDB
server with this dependency version.

The audit excludes these advisories only while no fix is available, using
`--ignore-until-fixed`. GitHub alerts remain open. Reassess these exceptions
before adding remote ChromaDB support. Reviewed on November 16, 2026.
