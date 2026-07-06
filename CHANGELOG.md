# Changelog

## [0.3.2](https://github.com/jorgenosberg/docstra/compare/docstra-v0.3.1...docstra-v0.3.2) (2026-07-06)


### Features

* **ci:** add GitHub Action for incremental docs updates ([b6e1c8f](https://github.com/jorgenosberg/docstra/commit/b6e1c8f0c9a875a5c29fdb54146bd12382cac702))
* **docs:** emit llms.txt and llms-full.txt with the site build ([9ee1c14](https://github.com/jorgenosberg/docstra/commit/9ee1c14304e29514d63e6df26f4378d781b2e080))
* **llm:** refresh model defaults and add shared client factory ([af561b6](https://github.com/jorgenosberg/docstra/commit/af561b6dfaf51536e36f259f15f19f1ba93f34f6))
* **llm:** support OpenAI-compatible local servers via api_base ([db507da](https://github.com/jorgenosberg/docstra/commit/db507da8d52b9cd781b7dccde6ed0c5238ce6285))
* **mcp:** serve the code index to agents over MCP ([dd35cab](https://github.com/jorgenosberg/docstra/commit/dd35cab9a1ed8be88d34883c1f55c2e7e8aab6f4))


### Bug Fixes

* **collector:** exclude dependency, cache, and build directories by default ([2f73524](https://github.com/jorgenosberg/docstra/commit/2f73524436998acac9b62e470e499b38bb010250))
* **ingestion:** survive oversized embeddings, duplicate chunk ids, and chroma telemetry noise ([84190e5](https://github.com/jorgenosberg/docstra/commit/84190e5b455fec895680e5b5c36d5332fb7bf2aa))

## [0.3.1](https://github.com/jorgenosberg/docstra/compare/docstra-v0.3.0...docstra-v0.3.1) (2026-07-06)


### Features

* **docs:** add deterministic documentation checks and check-docs command ([d172a4f](https://github.com/jorgenosberg/docstra/commit/d172a4f94bb8e4f7f1f0dd30dd6f6256ae58e2b9))
* **docs:** make docstra update regenerate only graph-impacted pages ([ff1366a](https://github.com/jorgenosberg/docstra/commit/ff1366af9e7c4005a2d405f4af27120f7a62582c))

## [0.3.0](https://github.com/jorgenosberg/docstra/compare/docstra-v0.2.0...docstra-v0.3.0) (2026-07-03)


### Features

* **cli:** add index command to build the core index without embeddings ([e637a58](https://github.com/jorgenosberg/docstra/commit/e637a581567ba6b06727d253ac687578a2e8e072))
* **docs:** derive cross-references from the import graph instead of embedding similarity ([7decb50](https://github.com/jorgenosberg/docstra/commit/7decb502d10347fa7cbc6ccb10cc9f1b05f4893a))
* **indexing:** expose graph-verified file cross-references on CodebaseIndex ([7c8a858](https://github.com/jorgenosberg/docstra/commit/7c8a858df47e30fe214a10efa422485813026183))

## [0.2.0](https://github.com/jorgenosberg/docstra/compare/docstra-v0.1.15...docstra-v0.2.0) (2026-07-03)


### Bug Fixes

* **ci:** read release PR branch from release-please output ([4132b9d](https://github.com/jorgenosberg/docstra/commit/4132b9d5cd77cddd331ebbf31c3a610b491c5e35))


### Miscellaneous Chores

* release 0.2.0 ([7ecb9f5](https://github.com/jorgenosberg/docstra/commit/7ecb9f5a60f81d4b245598c6c7f0d619a3c6ca72))
