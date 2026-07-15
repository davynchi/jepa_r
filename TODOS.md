# TODOs

## Choose repository ownership before publication

**What:** Decide whether this project remains a fork of `facebookresearch/jepa` or moves to a new repository, then update the Git remote and project metadata.

**Why:** The current `origin` points to upstream. Publishing or pushing without an ownership decision risks targeting the wrong repository and producing incorrect public links.

**Pros:** Clear ownership, safe push/release flow, and correct attribution.

**Cons:** Requires a repository name, GitHub owner, and external repository creation.

**Context:** Local v1 implementation is intentionally unblocked. Preserve the upstream license and attribute any code actually reused. Resolve this before the first public push or release.

**Depends on / blocked by:** Project owner's fork-versus-new-repository decision and GitHub access.

## Add the first real time-series benchmark

**What:** Select one concrete real-world time-series dataset and add a dataset adapter and benchmark protocol.

**Why:** Validate whether conclusions about SG/EMA extend beyond controlled synthetic dynamics.

**Pros:** External validity and a practical usage example.

**Cons:** Adds preprocessing, licensing, download/cache behavior, and new experimental controls.

**Context:** V1 deliberately avoids an abstract dataset framework. Choose the dataset first; introduce shared interfaces only after synthetic and real datasets expose a second concrete use case.

**Depends on / blocked by:** The synthetic v1 metrics/artifact contract is now stable; the
remaining blocker is selecting a concrete dataset and confirming its license.
