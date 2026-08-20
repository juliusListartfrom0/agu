# TASK-0258 Module A Amendment 001 — receipt-bound mechanical proof

## Status and immutable parent

This is a proposed amendment. It is not approved and authorizes no execution.
The three approved parent specification files remain byte-identical:

- `requirement.md`: `5c455408bd9ba730470b66189316578dc935b39a5cdc550c67e2a9c08d3f80d3`
- `solution.md`: `87f9a447112de8d294ffedbb1adfadb478713d94e00cdcf2e4f98999ea8a857d`
- `gate-review.md`: `cc52d809f42abba25ccdfb6667325612ce2f8961842f0e375db060ea2b70fa8f`

Approval of this amendment's exact SHA-256 would authorize implementation of
the amended Module-A proof boundary only. It would not authorize a model run.
A concrete versioned output root may be executed only after the amended
implementation has a different fresh-context review with Critical/Required
`0/0` and the user then separately authorizes that exact rerun. The immutable
v1 terminal evidence may never be changed. Module B remains unauthorized.

## Defect discovered during implementation review

The parent specification requires immutable `final_v1/mechanical_gate.json` to
state that atomic publication and absence of staging residue have passed. That
file must be finalized before its containing directory is published, so it
cannot truthfully prove the future publication of itself. The parent public
verifier also has no receipt-bound artifact for the required second empty-state
Swin extraction. Accepting booleans from the embedded gate would be a
self-attestation gap.

The current implementation therefore remains fail closed. Until this amendment
is approved and implemented, it may publish only `terminal_failure_v1`; it may
not publish a trusted success generation.

## Versioned state machine

An amended execution uses a new user-approved output root ending in
`vru_causal_temporal_retrospective_v2`. It does not reuse, rename, overwrite, or
repair v1. The frozen plan remains an externally receipted, read-only input
outside this root. The root contains its admission, zero to two recoverable
producer attempts, and these phase-specific targets:

```text
attempts/attempt-000N/          # parent producer resume contract, unchanged
candidate_v2/                   # immutable, explicitly nonterminal candidate
terminal_failure_v2/            # terminal failure before candidate publication
verified_result_v2/             # terminal verified result after candidate
postverification_failure_v2/    # terminal failure after candidate
```

The only trusted stable-directory states are:

1. `producer_incomplete`: none of the four phase targets exists; zero to two
   recoverable producer attempt directories may exist;
2. `pre_candidate_terminal_failure`: only `terminal_failure_v2` exists;
3. `candidate_verification_pending`: only `candidate_v2` exists;
4. `verified_terminal_result`: `candidate_v2` and only
   `verified_result_v2` exist; or
5. `postverification_terminal_failure`: `candidate_v2` and only
   `postverification_failure_v2` exist.

The supervisor may occupy the initial or verification-pending state while work
is active; neither state is terminal and neither yields a result capability.
No other combination is valid. Random `.task0258-*` staging is always untrusted
and must be absent before a result capability is minted.

`candidate_v2` is allowed to become visible before post-publication
verification because every eligibility flag is false and its gate schema has
no terminal decision. It grants no Module-B or rerun authority.

## Receipt-bound authorization capabilities

Prose approval is not sufficient at runtime. Before any implementation edit,
the human/orchestration boundary must record the user's approval as an
externally file-receipted exact artifact; the loader described below is the
implementation that later replays that already-recorded approval and is not a
prerequisite binary paradox. The approval schema is
`agu.task0258-module-a-amendment-implementation-approval.v1` with exactly:

```text
schema_version
module_id = existing-45-temporal-retrospective
repository_root_absolute_path
repository_root_device
repository_root_inode
parent_spec_approval_receipt
approved_parent_file_receipts
approved_amendment_file_receipt
amendment_fresh_review_receipt
implementation_scope_baseline_receipt
approval_scope = amendment_implementation_only
model_execution_authorized = false
module_b_authorized = false
approval_statement_sha256
approved_at_utc
artifact_sha256
```

The parent and amendment paths, byte sizes, file SHA-256 values, and fresh
review Critical/Required `0/0` result are exact. Before issuing this approval,
the external boundary seals the complete implementation-scope baseline defined
below; its independent internal/file receipt is
`implementation_scope_baseline_receipt`. This capability authorizes only the
closed code and test delta below. It can never authorize extraction or
evaluation.

After implementation, a different fresh review is sealed as
`agu.task0258-module-a-amended-implementation-review.v1` with exactly:

```text
schema_version
module_id = existing-45-temporal-retrospective
repository_root_absolute_path
repository_root_device
repository_root_inode
amendment_implementation_approval_receipt
implementation_context_id
reviewer_context_id
reviewer_independence = different_fresh_context
ordered_code_file_receipts
ordered_test_file_receipts
ordered_runtime_dependency_file_receipts
ordered_check_configuration_file_receipts
ordered_check_input_receipt_sets
implementation_scope_baseline_receipt
implementation_scope_delta
bootstrap_launcher_receipt
ordered_check_receipts
review_resource_summary
critical_count = 0
required_count = 0
optional_count
heavy_execution_performed = false
reviewed_at_utc
artifact_sha256
```

Context IDs are distinct nonempty safe slugs; this is a reviewed provenance
assertion, not a cryptographic identity proof. Code, test, and runtime-dependency
rows are ordered by repository-relative POSIX path and each has exactly `path`,
`size_bytes`, and `file_sha256`. The baseline receipt byte-equals the approval
field. `implementation_scope_delta` has the exact closed shape and equality
rules defined below. `ordered_check_receipts` has exactly, in this order,
`focused_pytest`, `full_pytest`, `ruff_check`, `ruff_format_check`,
`diff_check`, and `fresh_context_code_review`. Each row has exactly
`check_name`, `execution_protocol`, `sandbox_attestation_sha256`,
`command_sha256`, `exit_code=0`, `output_path`,
`output_size_bytes`, `output_artifact_sha256`, `output_file_sha256`, and
`completed_at_utc`.
`output_path` is the absolute no-symlink path to the fixed immutable final file
`<approved-check-output-directory>/<check_name>.out`; the six legal basenames
are derived only from the six fixed check names. The directory identity is
frozen in the implementation-scope baseline below, is outside the repository,
runtime image/contract, Module-A output, registry and bundle parents, and is
empty at approval. It may contain only those six finals plus at most one
transaction-bound stage named `<check_name>.<64-lowercase-hex-nonce>.stage`
while the trusted external runner publishes the
corresponding final. Size is a nonnegative integer. The first four command hashes are
computed from compact canonical JSON of exactly
`{launcher_argv,target_module,target_argv}`. `launcher_argv` is always
`[PYTHON,"-P","-S","/dev/fd/203","--task0258-review-request-fd","202","--task0258-source-fd","203"]`,
where `PYTHON` is the regular interpreter inside the retained runtime image.
`target_module` is `pytest` for the first two rows and `ruff` for the next two.
Finals are published in the exact six-check order. Before check ordinal `N`, the
directory contains exactly the first `N-1` finals and no stage; after success it
contains exactly the first `N` finals. A crash residue or unexpected child is a
fail-closed review state and is never deleted/repaired under this protocol.
`target_argv` is the corresponding parent requirement token array after
removing its leading `.venv/bin/python -m <module>`; both pytest arrays add the
exact tokens `-p`, `no:cacheprovider`, `--import-mode=importlib` immediately
after `-q`. Receipt-bound pytest configuration may not override that mode; the
bootstrap replays exact `sys.path` and terminal-finder state after collection,
after fixtures and before exit. Ruff and format
also add this exact argument immediately after
`scripts/screen_vru_causal_temporal_retrospective.py`:

```text
  scripts/task0258_module_a_verified_bootstrap.py \
```

Argument order is otherwise identical. This amendment intentionally replaces
the four path-backed parent review launchers while preserving their test/Ruff
targets; approval of this amendment's exact SHA approves those closed launcher
and cache/import-mode deltas. The trusted runner reserves stdin as a read-only
`/dev/null` descriptor, opens the one verified immutable bootstrap-source copy
on FD 203 at offset zero, and sends a compact-canonical
`agu.task0258-review-driver-request.v1` to FD 202. That request has exactly
`schema_version`, `module_id`, `review_execution_kind`, `check_name`,
`target_module`, `target_argv`,
`runtime_snapshot_receipt`, `namespace_provider_manifest_receipt`,
`expected_runtime_read_receipts`, `review_bootstrap_source_size_bytes`,
`review_bootstrap_source_sha256`, and `artifact_sha256`. Every non-artifact
value equals the corresponding launch-attestation value.
`review_execution_kind` is `discovery` or `evidence`;
namespace-provider/expected-read fields are null on initial discovery, bind the
current candidate fixed-point manifest on later discovery, and are non-null
exact arrays for evidence. Both phases use this same driver. The bootstrap, running before `site`, constructs `sys.path`
from the runtime manifest, treats `.pth` as inert bytes, installs the terminal
snapshot repository finder, and invokes only `runpy.run_module(target_module,
run_name="__main__", alter_sys=True)` with the fixed target argv. An unknown
module/argument, path-backed target, `site.main()`/`addsitedir` call, `.pth` execution, source/FD
mismatch, or second dispatch rejects. The fifth command hash is SHA-256 of
the literal protocol identifier `task0258-fixed-whitespace-v1`; it does not run
Git or a shell. The trusted runner applies one fixed byte algorithm to the
eighteen lexically ordered paths listed later: each must be a nonempty regular
non-symlink UTF-8 file, end in exactly one LF, contain no CR, trailing space or
tab, blank whitespace-only line, NUL, or line beginning with an unresolved
conflict marker `<<<<<<<`, `=======`, or `>>>>>>>`. This whole-file check covers
tracked, staged and untracked current bytes without consulting `.git`, index,
HEAD, attributes, filters, external diff or any Git configuration. Missing,
extra, reordered, empty, binary or invalid bytes fail. No caller-selected
command, policy or path is accepted.
For the first five rows, `execution_protocol` is exactly
`task0258-review-snapshot-fd-v2` and `sandbox_attestation_sha256` is a lowercase
SHA-256 of the exact attestation line embedded as the first output line. For
`fresh_context_code_review`, `execution_protocol` is exactly
`fresh-context-read-only-code-review-v1` and
`sandbox_attestation_sha256=null` because that evidence is the separately
sealed review artifact rather than a local executable check.
The fresh-review command hash is SHA-256 of the literal protocol identifier
`fresh-context-read-only-code-review-v1`; its output is a separate exact review
artifact naming the reviewed code/test/runtime/configuration/check-input
receipts and Critical/Required
`0/0`. Its `output_artifact_sha256` is non-null and equals that artifact;
the other five command outputs are raw logs and require this field null. The
full-suite receipt may not be replaced by a claim about
unrelated dirty-worktree failures. The review artifact is externally
file-receipted by the later authorization and is not itself an approval.

That separate output has schema
`agu.task0258-module-a-fresh-code-review.v1` and exactly
`schema_version`, `module_id`, `repository_root_absolute_path`,
`repository_root_device`, `repository_root_inode`,
`amendment_implementation_approval_receipt`,
`implementation_context_id`,
`reviewer_context_id`, `reviewer_independence=different_fresh_context`,
`ordered_code_file_receipts`, `ordered_test_file_receipts`,
`ordered_runtime_dependency_file_receipts`,
`ordered_check_configuration_file_receipts`,
`ordered_check_input_receipt_sets`, `implementation_scope_baseline_receipt`,
`implementation_scope_delta`, `bootstrap_launcher_receipt`,
`ordered_pre_review_check_receipts`,
`fresh_review_governance_observation`,
`critical_count=0`, `required_count=0`, `optional_count`,
`heavy_execution_performed=false`, `reviewed_at_utc`, and `artifact_sha256`.
Context IDs are distinct safe slugs; repository root, the five receipt lists,
baseline receipt, delta and bootstrap receipt exactly equal the
implementation-review values. The
approval receipt is the exact current `ArtifactFileReceipt`.
`ordered_pre_review_check_receipts` is exactly the first five
`ReviewedCheckReceipt` rows from the implementation-review artifact, in order,
so the fresh reviewer attests that it inspected the current canonical command
evidence rather than an arbitrary/stale green claim. Counts are non-boolean
nonnegative integers; timestamp is UTC RFC3339. Its internal/file receipt must
equal the `fresh_context_code_review` row's path/size/internal/file hashes. The
implementation-review loader replays the current approval, all five check
outputs, this artifact, and their semantic equality, not only file hashes.
`fresh_review_governance_observation` is the exact
`FreshReviewGovernanceObservation` defined below. The implementation review's `review_resource_summary` is exactly
`{runtime_build_observation,ordered_check_observations,fresh_review_observation}`;
`runtime_build_observation` is exactly
`{build_resource_limits,build_resource_observation,postpublication_free_bytes}`,
where the middle value is the contract's exact
`RuntimeBuildResourceObservation` field named
`build_prepublication_observation`. It comes
from the verified runtime snapshot contract/receipt pair;
each of its five check rows is exactly
`{process_resource_observation,output_publication_observation}` and equals the
corresponding verified execution attestation plus independently recomputed
published output sizes. The fresh value is the exact
`FreshReviewResourceSummary` whose governance row equals the artifact's
`fresh_review_governance_observation` and whose size equals its externally
verified file size. `heavy_execution_performed=false`
is derived only when all observations are within their frozen limits and every
production-worker/media-or-checkpoint/device-or-model-service/embedding-
publication audit counter is zero under the exact definition below; otherwise
neither review artifact may be sealed. No claim about the count of CPU-only
`weights=None` constructors is made.

The reviewed implementation closure is exact, not caller-selected. Code rows
must be these five paths in lexical order:

```text
app/analysis/vru_causal_temporal_retrospective.py
scripts/extract_vru_causal_tiled_swin_embeddings.py
scripts/screen_vru_causal_temporal_retrospective.py
scripts/seal_vru_causal_temporal_feature_plan.py
scripts/task0258_module_a_verified_bootstrap.py
```

Adding only the bootstrap path is an explicit amendment to the parent's
four-code-file implementation scope and canonical Ruff/format/diff commands.
No other implementation or test path is added. Approval of this amendment's
exact SHA is therefore required before that file may be created.

Test rows must be these five paths in lexical order:

```text
tests/test_task0258_module_a_cli.py
tests/test_vru_causal_final_evaluator.py
tests/test_vru_causal_temporal_feature_plan.py
tests/test_vru_causal_temporal_retrospective.py
tests/test_vru_causal_tiled_swin_embeddings.py
```

The implementation approval also freezes the pre-edit artifact
`agu.task0258-module-a-implementation-scope-baseline.v1`. It has exactly
`schema_version`, `module_id`, `repository_root_absolute_path`,
`repository_root_device`, `repository_root_inode`, `ordered_root_paths`,
`check_output_directory_absolute_path`, `check_output_directory_device`,
`check_output_directory_inode`,
`ordered_entry_receipts`,
`ordered_repository_executable_receipts`,
`captured_at_utc`, and `artifact_sha256`. `ordered_root_paths` is exactly `[.]`.
The external issuer walks the entire repository root no-follow before any
approved edit, excluding only the two exact root children `.git` and `.venv`,
and records every other root and descendant, including non-executable data and
the absent future bootstrap path. Excluded-root names are schema constants, not
caller input; any third exclusion rejects. The separately selected check-output
directory is a pre-existing empty no-symlink directory with mode `0o700`; its
absolute path and physical identity are frozen by the three exact fields above.
It is disjoint by path, casefold path, ancestry, device/inode and hardlink group
from every repository/runtime/Module-A input or output. Each
`ImplementationTreeEntryReceipt` is exactly
`{path,entry_kind,mode_bits,link_count,size_bytes,file_sha256,symlink_target_text,hardlink_group_sha256,ordered_child_names}`.
`path` is repository-relative and lexical; `entry_kind` is `absent`,
`directory`, `excluded_directory`, `regular`, or `symlink`; mode is a non-boolean integer. A regular
row has `link_count=1`, size/file hash and the hash of its singleton lexical
path array, with the other conditional fields null. Any scope regular file with
`st_nlink != 1` fails; this detects an alias created anywhere on the physical
filesystem without needing a root-limited alias search. A directory has
only the exact lexical child-basename array. A symlink has only its exact target
text and is never followed. Exactly `.git` and `.venv` have
`entry_kind=excluded_directory`, are verified as real no-follow directories,
retain mode while descendant/member fields are null, and are never eligible as
review/runtime query inputs; no other path may use that kind. The root directory
child array still includes both names. The one absent bootstrap row has every conditional
field null. Unknown filesystem kinds, duplicate/case-fold-colliding paths,
walk races, path escape, or bytes/metadata changing between two complete walks
fail approval rather than producing a baseline.

The repository root is an externally selected pre-existing no-symlink
directory. Its absolute lexical path and no-follow physical `{device,inode}`
are frozen first in the baseline and then repeated byte-for-byte in the
implementation approval, implementation review, fresh review, rerun
authorization, every review launch attestation, and every absolute path map.
Device and inode are non-boolean nonnegative integers.
Every relative receipt resolves beneath that one identity. A byte-identical
clone, renamed replacement, different root, or cross-root mixture rejects even
when all leaf hashes match.

`ordered_repository_executable_receipts` is the lexical executable subset
projected from the already complete `ordered_entry_receipts`, not a second
authority or an open-ended discovery scan. It includes every regular or symlink
path whose basename ends in `.py`, `.pyi`,
`.pyx`, `.sh`, or `.zsh`, every `conftest.py`, and every regular file with any
execute bit. Regular rows are exact `ReviewedFileReceipt` plus `mode_bits` and
`link_count=1`; symlink rows are exact path/mode/target and cause approval to
fail if they resolve to executable local code. The implementation review and
every later loader require the same path set, modes, bytes and singleton link
counts outside the ten authorized leaf rows. More importantly, the complete
entry array requires the same equality for every executable or non-executable
repository member, so a new/modified JSON, model-config, fixture or other local
data file cannot become authorized merely by appearing in a new check-input or
runtime read closure.

`implementation_scope_delta` is exactly
`{ordered_leaf_rows,ordered_directory_rows}`. `ordered_leaf_rows` has exactly
ten rows in the combined lexical order of the five code and five test paths.
Each row is exactly
`{path,before_entry,after_file,change_kind}`. `before_entry` is the exact
baseline row; `after_file` is the exact current `ReviewedFileReceipt`;
`change_kind` is `unchanged`, `modified`, or, only for the bootstrap path,
`created`. `ordered_directory_rows` has exactly one row,
`{path=scripts,before_child_names,after_child_names,authorized_added_children,authorized_removed_children}`.
Its before array equals the baseline `scripts` directory row; added children is
exactly `[task0258_module_a_verified_bootstrap.py]`; removed is empty; after is
the exact lexical before-plus-added array. Delete, rename, symlink, link-count
change, or another created path is forbidden. Under the same whole-repository no-follow root
walk, every baseline entry outside the ten leaf rows and this one authorized
directory-membership delta must retain identical membership, kind, mode,
link-count, bytes, symlink target, hardlink group and directory children. This comparison is performed
before every check, after every check, by the implementation-review loader,
the fresh reviewer and the rerun-authorization loader. Current runtime or test
discovery can add evidence inputs but can never expand edit authority or replace
the external pre-edit baseline. Every repository query observed by discovery or
evidence must resolve to a current row in this baseline plus the exact ten-leaf
delta; an absent-baseline path is a review failure rather than a new evidence
input. For non-authorized entries the contract intentionally proves content,
kind, mode, link and membership equivalence, not provenance of an inode: an
adversarial delete-and-byte-identical-recreate that preserves every frozen
semantic field is treated as equivalent and is outside the no-replacement
claim. Root replacement remains rejected by the separately frozen root identity.

`ordered_check_configuration_file_receipts` is exactly these two
`ReviewedFileReceipt` rows in lexical order:

```text
pyproject.toml
pytest.ini
```

Empty, missing, extra, duplicate, reordered, or renamed rows fail. The rerun
authorization and both authorization loaders require these same exact paths
and receipts; a caller cannot narrow the at-use drift boundary.

`ordered_check_input_receipt_sets` has exactly five rows, in the same order as
the five executable checks before fresh review. Every row is exactly
`{check_name,ordered_query_receipts,projection_sha256}`. A
`ReviewedNamespaceQueryReceipt` is exactly
`{operation,path,arguments,follow_policy,result_kind,errno,stat_result,access_result,readlink_target_text,ordered_directory_entries,xattr_result,file_content}`.
Operation is exactly `access`, `exec`, `fstat`, `getdents`, `lstat`, `mmap`, `open`,
`readlink`, `stat`, or `xattr`; path is repository-relative. `arguments` is the
closed operation-specific object: access has `{mode,follow_symlinks}`; stat has
`{follow_symlinks}`; open has `{flags,creation_mode}`; fstat and getdents have
`{source_open_query_sha256}`; mmap has
`{source_open_query_sha256,offset,length,access}`; xattr has
`{name,options,follow_symlinks}`;
and exec, lstat, and readlink use `{}`. Write/create/truncate flags on repository
paths are forbidden. `follow_policy` is exactly `follow`, `no_follow`, or
`not_applicable`: access/stat copy their requested boolean to `follow` or
`no_follow`; xattr copies its requested boolean the same way;
lstat/readlink are `no_follow`; open is `no_follow` iff its flags
contain the frozen `O_NOFOLLOW` bit and otherwise `follow`; exec is `follow`;
fstat/getdents/mmap are `not_applicable`. A caller may not invent a
follow-symlinks argument for an operation that has none.

`result_kind` is `success` or `error`. Error rows have one exact errno from
`[EACCES,ELOOP,ENOENT,ENOTDIR]` and every result field null. Success rows have
null errno and exactly the operation-specific applicable result projection or
projections defined below. `stat_result` is
exactly
`{entry_kind,mode,device,inode,nlink,uid,gid,rdev,size_bytes,atime_ns,mtime_ns,ctime_ns,birthtime_ns,flags}`
with non-boolean integers and entry kind `directory`, `regular`, or `symlink`;
stat, lstat and fstat remain separate queries and may have different results. Access
has one exact boolean. Readlink has exact target text. Getdents has the lexical
array of exact `{name,entry_kind,inode}` rows. Xattr has
`{name,size_bytes,value_sha256}`. A successful open always records the full
fstat projection; when it opened a directory with the frozen read-only
`O_DIRECTORY` flags, `file_content=null` and directory membership appears only
in a separate getdents query. A successful regular-file open, mmap, or exec has
exact `file_content={size_bytes,file_sha256}` and its corresponding full stat
projection. mmap/exec of a non-regular object rejects. Inapplicable result
fields are null. Unknown flags, errno, fields, coercions,
duplicate query keys or two different results for the same immutable query
fail.

Rows are sorted by compact-canonical `(operation,path,arguments,follow_policy)`,
nonempty and duplicate-free; their compact-canonical array hash equals
`projection_sha256`. They are the complete immutable namespace-query manifest
produced by the fixed-point trusted runner below, including every successful
or failed repository path resolution by the entire evidence process tree, not
only opened files. A static closure is only a mandatory seed and can never
narrow the audited set. Every mandatory parent/amendment/code/test/runtime/
configuration seed content query must byte-equal its separately reviewed
receipt. The evidence root is an attested read-only namespace filesystem whose
provider replays the recorded stat/lstat device/inode/timestamps, access result,
directory entries, symlink target, xattr and bytes exactly; unlisted or changed
queries fail. A plain copied directory whose kernel metadata differs is not a
valid evidence root. The implementation-review loader rejects substituted query
results even if the original path was later restored. If the host cannot attest
the namespace provider and exact query replay, the check fails.

For `focused_pytest`, roots are the five exact test files above, every ancestor
`conftest.py` beneath the receipt-bound repository root, every local pytest
plugin named by the receipt-bound configuration, and their complete recursive
repository-local Python import closure. For `full_pytest`, roots are every
regular non-symlink file beneath the exact receipt-bound `testpaths=tests` that
matches the frozen pytest-version default patterns `test_*.py` and
`*_test.py` (the configuration must not override `python_files`), plus every applicable
ancestor `conftest.py`, local pytest plugin, and their complete recursive local
import closure are mandatory seeds. Resolution parses AST without executing it. A literal
repository-local `importlib.import_module`, `__import__`, pytest plugin string,
or file-based module load is resolved and included; a nonliteral local dynamic
load does not become trusted merely because static analysis misses it: the
kernel audit must observe it and the evidence sandbox must serve it from the
snapshot. Ambiguous namespace, missing source, audit overflow, symlinked
matching test or unaudited repository read fails. A new, removed, renamed,
dynamically loaded or newly collected test/helper/plugin/data fixture/subprocess
script changes the exact set and rejects. The full-suite set may contain more
than the five edited test rows; that is check evidence, not edit authority.

For `ruff_check` and `ruff_format_check`, the set is exactly the five code and
five edited test rows plus the two check-configuration rows. For `diff_check`,
it is exactly these eighteen literal paths in lexical order:

```text
app/analysis/vru_causal_temporal_retrospective.py
docs/specs/TASK-0258-temporal-canary/amendment-001-postpublication-proof.md
docs/specs/TASK-0258-temporal-canary/capability-map.md
docs/specs/TASK-0258-temporal-canary/code-review.md
docs/specs/TASK-0258-temporal-canary/development.md
docs/specs/TASK-0258-temporal-canary/gate-review.md
docs/specs/TASK-0258-temporal-canary/requirement.md
docs/specs/TASK-0258-temporal-canary/solution.md
docs/specs/TASK-0258-temporal-canary/testing.md
scripts/extract_vru_causal_tiled_swin_embeddings.py
scripts/screen_vru_causal_temporal_retrospective.py
scripts/seal_vru_causal_temporal_feature_plan.py
scripts/task0258_module_a_verified_bootstrap.py
tests/test_task0258_module_a_cli.py
tests/test_vru_causal_final_evaluator.py
tests/test_vru_causal_temporal_feature_plan.py
tests/test_vru_causal_temporal_retrospective.py
tests/test_vru_causal_tiled_swin_embeddings.py
```

Current regular-file bytes are mandatory snapshot seeds in that order; no Git
metadata is an input. The implementation review, fresh review, rerun
authorization and every at-use loader require the same complete per-check
snapshot sets. An exit-zero log whose immutable executable/data input snapshot
is absent, stale, incomplete or different is not evidence.

`ordered_runtime_dependency_file_receipts` is a separate exact lexical list of
the complete repository-local executable import closure. Starting from the
five code paths above, parse Python AST without importing; resolve every
absolute or relative `app.*`, `scripts.*`, and `utils.*`
`Import`/`ImportFrom` to its one repository `.py` file, add it, and recurse to a
fixed point. The closed repository package-root table is exactly
`[app,scripts,utils]`; include every package `__init__.py` traversed by
resolution. A repository-local import outside that table, namespace ambiguity,
missing local source, import from outside the repository, namespace package, or
symlink fails. Outside the exact bootstrap file, any use of `__import__`, `importlib`,
`compile`, `exec`, `eval`, runtime mutation of `sys.path`, or another dynamic
local-code loader in that closure fails review. The bootstrap has one closed
exception: it may import only `importlib.abc` and `importlib.util`, define one
`SnapshotRepositoryFinder` and one `SnapshotRepositoryLoader`, call
`importlib.util.spec_from_loader` only from that finder, and call
`compile(snapshot_bytes, virtual_filename, "exec", dont_inherit=True,
optimize=0)` followed by `exec(code, module.__dict__)` only inside that loader's
`exec_module`. The finder accepts only the exact module-name-to-immutable-bytes
map produced by the verified closure snapshot; its `find_spec` returns `None`
for every other name, its loader performs no filesystem read, and the virtual
filename is the frozen repository-relative receipt path. The bootstrap may not
call `eval`, `__import__`, `SourceFileLoader`, `SourcelessFileLoader`, mutate
`sys.path` after installing the verified snapshot root, or use any other
dynamic loader. One disjoint review-driver branch may import stdlib `runpy` and
call `runpy.run_module` exactly once for installed-runtime module `pytest` or
`ruff` with the fixed target argv above; it cannot name/load a repository
module, run in a production operation, or return a production capability. The
AST scanner recognizes these exceptions only in the exact
bootstrap path and exact class/method/call sites; the same node in any other
path or function fails review.
The list must include at least
`app/analysis/vru_causal_training_v2.py` and
`app/analysis/vru_causal_video_probe_v2.py`; those are examples, not an
allowlist. Standard-library and installed-distribution dependencies retain only
the parent's exact interpreter/package versions, `show_config`, and build-info
hashes; those are environment observations, not wheel/source-file receipts and
do not claim to detect malicious same-version package replacement. At use, the loader
recomputes the local closure using the receipt-bound scanner implementation,
requires exact path equality, and reopens every file no-follow against this
list. Edited-file scope and executable dependency scope are therefore distinct.

Only a subsequent explicit user instruction may be sealed as the externally
file-receipted exact artifact
`agu.task0258-module-a-v2-rerun-authorization.v1`, with exactly:

```text
schema_version
module_id = existing-45-temporal-retrospective
repository_root_absolute_path
repository_root_device
repository_root_inode
amendment_implementation_approval_receipt
implementation_review_receipt
approved_amendment_file_receipt
approved_code_receipts
approved_test_receipts
approved_runtime_dependency_receipts
approved_check_configuration_receipts
approved_check_input_receipt_sets
approved_check_receipts
approved_implementation_scope_baseline_receipt
approved_implementation_scope_delta
approved_bootstrap_launcher_receipt
approved_static_input_contract
allowed_operations
output_root_absolute_path
candidate_receipt_bundle_absolute_path
run_id
authorization_scope = one_module_a_v2_rerun
maximum_run_count = 1
run_admission_relative_path = run_admission.json
run_consumption_registry_directory_absolute_path
run_consumption_registry_directory_identity
module_b_authorized = false
approval_statement_sha256
approved_at_utc
artifact_sha256
```

`allowed_operations` is the exact ordered array
`[preflight_and_publish_admission,recover_admission_completion,publish_candidate,seal_candidate_receipt_bundle,postpublication_verify,load_existing_terminal]`.
It authorizes the one phased workflow, not six runs; an invocation selects
exactly one member and the run-history state machine determines whether that
operation is currently legal. No operation may be added, omitted, reordered,
renamed, repeated outside its legal state, or selected from caller-controlled
module/callable text.

The rerun authorization's repository-root triple exactly equals the approved
baseline, implementation approval, implementation review and fresh-review
triples. Every loader reopens that exact directory no-follow and compares its
physical identity before resolving any relative path or reading any project
file; it never derives a new root from cwd or a supplied leaf path.

`approved_static_input_contract` is exactly
`{temporal_plan_artifact_sha256,temporal_plan_file_sha256,task0257_receipts_projection_sha256}`.
The TASK-0257 projection is compact canonical JSON of the complete exact
`Task0257ExpectedReceipts` value defined by the parent, including every ordered
JPEG, source-video, and checkpoint receipt; it is not a caller-selected subset.
Its top-level object has exactly the parent dataclass field names; tuple fields
become JSON arrays in their declared order; every leaf is exactly the parent's
six-field `StoredArtifactReceipt` or three-field `FileReceipt`; there are no
class/type tags, paths, omitted fields, or coercions. Compact canonical JSON is
UTF-8 with sorted object keys, `(',', ':')` separators, no NaN/Infinity, and one
final LF before hashing.
All three values are lowercase SHA-256 strings. The candidate-receipt bundle
path is an absent, absolute, no-symlink file outside the output root and the
consumption registry. `run_consumption_registry_directory_identity` is exactly
`{device,inode}` for the already-existing registry directory, frozen by the
external authorization issuer. The output root must be the absent, no-symlink,
absolute
`vru_causal_temporal_retrospective_v2` root named in the user's instruction;
`run_id` is a nonempty safe slug. The consumption registry is a pre-existing,
user-approved, no-symlink directory outside the output root whose contents are
owned by this authorization boundary and retained after the output root moves
or disappears. The registry transaction and the one authorization-bound
candidate-bundle transaction are the only two explicit amendments to the
parent's "write only the output root" rule. Only the fixed claim/completion files and
bounded monotone history markers defined below may be created there. Together
they are capped at `16_777_216` bytes. They use mode `0o600`, no-follow opens,
canonical bytes, file/stage/registry-directory fsync, no-clobber rename, exact
ancestor/casefold/inode alias guards, and the same parent lock discipline.

### Exact nested receipt vocabulary

Every nested receipt is closed and role-specific:

- `ArtifactFileReceipt` is exactly `{artifact_sha256,file_sha256}`. The name
  `HistoryArtifactReceipt` elsewhere in this amendment is the same exact shape
  restricted to the run-history graph; it adds no field. Parent approval,
  amendment approval, implementation review, rerun authorization, run
  admission, candidate bundle, and result/failure JSON receipts use this shape.
- `NamedFileReceipt` is exactly `{filename,size_bytes,file_sha256}` with one
  safe basename. Parent approved spec files use it.
- `ReviewedFileReceipt` is exactly `{path,size_bytes,file_sha256}` with one safe
  repository-relative POSIX path. Amendment/code/test/runtime-dependency rows
  use it.
- `RuntimeTreeMemberReceipt` is exactly
  `{root_id,relative_path,entry_kind,mode_bits,link_count,size_bytes,file_sha256,symlink_target_text,resolved_target_root_id,resolved_target_relative_path,hardlink_group_members,hardlink_group_sha256}`.
  Root ID is `venv_site_packages`, `venv`, `base_runtime`,
  `bootstrap_source`, or `subprocess_tool`; paths are safe relative POSIX
  paths. Directory rows have
  only mode. Regular rows have mode, a positive non-boolean link count, size,
  file hash, the lexical duplicate-free array of every exact in-image
  `{root_id,relative_path}` sharing their source inode, and SHA-256 of the
  compact-canonical group array. Every member of one group repeats the identical
  array/hash, has the same source device/inode and bytes, and
  `link_count == len(hardlink_group_members)`; a source inode with any
  root-external link rejects. Symlink rows have mode, exact target text and an
  in-image resolved root/path; link/group fields are null.
  Every symlink resolves within the sealed image, loops/escapes fail, and the
  interpreter row itself is regular. The array is lexical by root then path,
  duplicate-free and exact-coverage. Its physical image location is always
  `<root_id>/<relative_path>`; root prefixes cannot alias or be caller-selected.
- `RuntimeSnapshotContract` is a standalone compact-canonical artifact with
  schema `agu.task0258-runtime-snapshot-contract.v1` and exactly
  `{schema_version,module_id,manifest_absolute_path,manifest_artifact_sha256,manifest_file_sha256,runtime_root_absolute_path,runtime_root_device,runtime_root_inode,ordered_mount_flags,ordered_origin_root_rows,ordered_python_sys_path_entries,python_snapshot_location,python_executable_copy_receipt,bootstrap_source_copy_receipt,ordered_venv_executable_symlink_receipts,ordered_tree_member_receipts,tree_projection_sha256,ordered_subprocess_executable_receipts,review_heavy_execution_policy_receipt,os_system_runtime_receipt,build_resource_limits,build_prepublication_observation,artifact_sha256}`.
  `RuntimeSnapshotReceipt` is the small exact
  `{contract_absolute_path,artifact_sha256,file_sha256,postpublication_free_bytes}`
  for that standalone artifact. The contract path is absolute, no-symlink,
  outside the repository,
  output root, consumption registry and candidate-bundle parent; the contract
  file is immutable, mode `0o600`, at most `67_108_864` bytes, and independently
  frozen by the external review issuer. Every consumer first verifies this
  four-field receipt and then parses/replays the complete contract; its
  `postpublication_free_bytes` is measured on the one common filesystem device
  shared by the runtime root and contract parent. The two parents must have the
  same `st_dev`; cross-filesystem layout rejects before either stage is created.
  No large
  contract array is copied inline into a launch, output log, approval or review
  artifact.
  The tree projection is the compact-canonical hash of the exact member array;
  subprocess receipts use the separately frozen `ExecutableCopyReceipt` shape
  below. The retained root is an already finalized no-symlink directory mount
  whose exact tree is described by the manifest; its Python path is a regular
  member. Every loader reopens the manifest and complete tree no-follow and
  checks mount flags before use. `ordered_mount_flags` is exactly
  `[nodev,nosuid,read_only]`; execution is additionally constrained by the
  sandbox's executable allowlist. `python_snapshot_location` is exactly
  `{root_id=base_runtime,relative_path}` and must identify the copied regular
  resolved base interpreter. Its parent layout is the copied base-runtime
  prefix, not the `venv` root, and no adjacent `pyvenv.cfg` may redirect it.
  The separately copied `.venv/pyvenv.cfg` is inert evidence data; neither
  review nor production startup relies on its original absolute `home` value.
  There is no ambiguous member-relative string.
  `ordered_python_sys_path_entries` is exactly these three rows and this order:

  ```text
  {ordinal=1,root_id=base_runtime,relative_path=lib/python3.11,purpose=stdlib}
  {ordinal=2,root_id=base_runtime,relative_path=lib/python3.11/lib-dynload,purpose=dynload}
  {ordinal=3,root_id=venv_site_packages,relative_path="",purpose=site_packages}
  ```

  The empty relative path is allowed only in the third row and means the root
  itself. Every target is a directory represented by the contract tree. Review
  and production bootstrap replace `sys.path` with exactly these three physical
  image paths in this order; empty/CWD/user-site/original-repository paths,
  additions, omissions, reorderings and a same-name shadow module outside these
  three roots reject before target import.
  The manifest has schema `agu.task0258-runtime-snapshot-manifest.v1` and exactly
  `schema_version`, `module_id`, `ordered_mount_flags`,
  `ordered_origin_root_rows`, `ordered_python_sys_path_entries`,
  `python_snapshot_location`,
  `python_executable_copy_receipt`, `bootstrap_source_copy_receipt`,
  `ordered_venv_executable_symlink_receipts`,
  `ordered_tree_member_receipts`, `tree_projection_sha256`,
  `ordered_subprocess_executable_receipts`, `review_heavy_execution_policy_receipt`,
  `os_system_runtime_receipt`,
  `build_resource_limits`, `build_prepublication_observation`, and
  `artifact_sha256`. Its arrays
  byte-equal the corresponding enclosing receipt values and its independent internal/file
  hashes equal the manifest receipt. It is the fixed special root member
  `.task0258-runtime-snapshot-manifest.json`; `manifest_absolute_path` must equal
  `runtime_root_absolute_path` joined to that basename. To avoid a self-hash
  cycle, this one special member is excluded from `ordered_tree_member_receipts`
  and is covered only by the enclosing manifest internal/file receipt. No
  alternate manifest path or basename is accepted.
  `ordered_origin_root_rows` has exactly five rows in this precedence order,
  each shaped `{precedence,root_id,source_absolute_path,inclusion_rule}`:
  `venv_site_packages` is the resolved `.venv` site-packages subtree;
  `venv` is the complete `.venv` root excluding that first subtree and the
  exact executable-symlink basenames `bin/python`, `bin/python3`, and
  `bin/python3.11`; those three source links are captured only by
  `ordered_venv_executable_symlink_receipts` and are never recreated in the
  image. That array has exactly those three paths in lexical order and rows
  `{relative_path,target_text}`; the first row's resolution chain and final
  bytes equal `python_executable_copy_receipt.source_executable_receipt`;
  `base_runtime` is the resolved base-Python prefix excluding prior matches;
  `bootstrap_source` uses the approved repository's `scripts` directory as its
  source and includes only `task0258_module_a_verified_bootstrap.py`; and
  `subprocess_tool` uses source `/` but includes only the separately fixed
  non-system executable paths. The bootstrap source's exact
  `BootstrapSourceCopyReceipt` is
  `{source_reviewed_file_receipt,runtime_member_location,size_bytes,file_sha256}`;
  the source row is the approved `ReviewedFileReceipt`, location is exactly
  `{root_id=bootstrap_source,relative_path=task0258_module_a_verified_bootstrap.py}`,
  and size/hash equal both source and copied regular tree member. Precedence is the non-boolean integer
  `1..5`; source paths are absolute, no-follow stable roots. Every source member
  is assigned to the first matching row, appears exactly once, and its
  `relative_path` is relative to that row's source root. The exact-tool-set rule
  never acts as a broad `/` walk. OS-sealed frameworks are excluded from this
  copied-member namespace and represented only by the separate
  `os_system_runtime_receipt`. Overlap, no match, multiple assignment,
  a different order, or a caller-selected origin rejects.
  Its production member set comes from a static no-follow closure, never from
  the non-heavy review read subset: include every member of the resolved base
  Python distribution prefix, the `.venv` tree subject only to the three exact
  symlink exclusions above, every installed
  distribution/site-packages member (including `.pth` bytes but never executing
  `.pth` code), Torch/Torchvision package data and native
  libraries, Metal/MPS backend resources, and every fixed subprocess tool.
  `os_system_runtime_receipt` is exactly
  `{operating_system,build_version,kernel_version,architecture,ordered_code_signature_rows}`;
  signature rows are lexical exact `{absolute_path,team_identifier,cdhash}` for
  every allowed OS framework/tool, with no user-writable path. Directory/symlink membership is walked twice to stability and every regular
  byte is hashed. OS-sealed Apple system frameworks remain an explicit OS TCB
  identified by OS build/code-signature receipts; no user-writable system path
  is allowed. Missing/extra/drift relative to this full static tree rejects.
  Each review launch records the discovery-derived expected-read subset and its
  post-exit attestation records the independently audited observed subset; they
  must be exactly equal and every member must occur in the full tree. That
  subset is evidence of check inputs, not the definition or upper bound of
  production runtime coverage.
- `ReviewHeavyExecutionPolicyReceipt` is exactly
  `{policy_absolute_path,artifact_sha256,file_sha256}` for a standalone
  compact-canonical artifact with schema
  `agu.task0258-review-heavy-execution-policy.v1` and exactly
  `{schema_version,module_id,provider_receipt,ordered_process_role_rows,ordered_protected_path_rows,ordered_device_endpoint_rows,ordered_network_endpoint_rows,ordered_write_endpoint_rows,ordered_relevant_event_kinds,path_matching_protocol,process_tree_projection_protocol,event_projection_protocol,artifact_sha256}`.
  `provider_receipt` is exactly
  `{provider_protocol=task0258-kernel-process-audit-v1,provider_name,provider_version,provider_build_sha256,provider_code_signature}`;
  the code-signature value is exactly
  `{absolute_path,team_identifier,cdhash}` and must be one verified OS TCB row
  in the runtime contract. Process-role rows are one closed ordered set;
  pytest/Ruff runpy target code remains inside `review_driver`, not a
  fictitious second process.
  Each process-role row is exactly
  `{role,ordered_allowed_parent_roles,executable_receipt_kind,source_receipt_required}`.
  The four rows/order are exactly
  `{runtime_builder,[trusted_external_runner],provider_code_signature,false}`,
  `{review_driver,[trusted_external_runner],RuntimeExecutableFileCAS,true}`,
  `{synthetic_worker,[review_driver],RuntimeExecutableFileCAS,true}`, and
  `{fixed_subprocess_tool,[review_driver,synthetic_worker],RuntimeExecutableFileCAS,false}`.
  Protected-path rows are exactly
  `{subject_role,absolute_path,device,inode,size_bytes,file_sha256}` and have
  exactly these roles/order: `checkpoint`, `source_video_hazen`,
  `source_video_randolph`, `source_video_vtv`, and `source_video_harwood`.
  Values equal the independently approved input paths/receipts plus fresh
  no-follow physical identity. No two rows or any write/runtime/repository root
  may share path, casefold path, device/inode or a hardlink group; ancestors are
  safe directories and a non-unique inode rejects. Device rows are exact
  `{endpoint_role,normalized_endpoint,match_kind}` and exactly these rows/order:
  `{attested_review_source_fd,/dev/fd/203,allow_review_driver_open_as_script_only_with_RuntimeSourceFileCAS}`,
  `{attested_synthetic_source_fd,/dev/fd/4,allow_synthetic_worker_open_as_script_only_with_RuntimeSourceFileCAS}`,
  `{filesystem_device,/dev/*,deny_except_dev_null_and_urandom}`,
  `{mps_gpu_iokit,*,deny_all_iokit_gpu}`, and
  `{external_model_service,*,deny_all_mach_xpc_model}`. Endpoint rows are
  evaluated in this order. The two `/dev/fd` exceptions require the exact role,
  parent, script-open operation, inherited regular open-file description,
  offset zero and fresh source CAS; they authorize no other FD or ordinary
  device access. The network array is exactly one row `{policy=deny_all}`.
  Write rows are exact
  `{write_role,ordered_allowed_subject_roles,allowed_parent_absolute_path,allowed_parent_device,allowed_parent_inode,ordered_allowed_relative_prefixes,ordered_allowed_operations,maximum_bytes}`
  and have exactly these rows/order. `review_temp` allows
  `[trusted_external_runner,review_driver,synthetic_worker,fixed_subprocess_tool]`
  to use only `[audit/,synthetic/,outputs/,tmp/]` under the one attested temporary
  ancestor with operations
  `[create_new_regular,write_owned_regular,truncate_owned_regular,fsync,rename_within_prefix,unlink_owned_temporary]`
  and `maximum_bytes=67_108_864`. `runtime_image_stage` allows only
  `[runtime_builder]` to create/write/fsync the one content-addressed runtime
  tree and perform its final no-clobber rename under the attested image parent,
  with `maximum_bytes=4_294_967_296`. `runtime_contract_stage` allows only
  `[runtime_builder]` to create/write/fsync the one authorization-bound contract
  stage and perform its final no-clobber rename under the attested contract
  parent, with `maximum_bytes=67_108_864`. `check_output_stage` allows only
  `[trusted_external_runner]` to create/write/fsync the one nonce-bound stage
  with the exact basename grammar above,
  no-clobber rename it to the fixed `<check_name>.out`, fsync the approved
  check-output parent and reopen the final, with
  `maximum_bytes=16_777_216`. The two runtime parent devices must equal one
  another; the check-output parent is the identity frozen in the baseline.
  Subject roles are drawn only from the four process-role rows plus the external
  runner TCB literal. Any cross-role use, unlisted flag/operation, overwrite,
  final-file mutation or second stage rejects. Parent paths are existing
  no-symlink directories with fresh physical identity. Every other write,
  including any embedding or production output role, is denied. The policy is not asserted over the
  external fresh-context governance reviewer; that separately receipted review
  may use its service but may not run a local Module-A production worker or
  claim a local OS process observation. Relevant event
  kinds are exactly
  `production_worker_spawn`, `approved_media_or_checkpoint_open`,
  `device_or_model_service_access`, `embedding_publication`,
  `unknown_descendant`, and `unknown_relevant_event`. Path matching is exactly
  `no-follow-ancestor-device-inode-v1`, `process_tree_projection_protocol` is
  exactly `compact-canonical-ordered-process-rows-sha256-v1`, and
  `event_projection_protocol` is exactly
  `compact-canonical-ordered-relevant-events-sha256-v1`. The process projection is SHA-256 of the
  compact-canonical ordered process-row array; the event projection is SHA-256
  of the compact-canonical ordered relevant-event array; neither preimage has a
  final LF. Unknown/missing/extra/reordered policy rows reject.
  The independently externally receipted policy artifact binds the trusted
  runner's OS sandbox/audit policy, not a Python callback.
  Its path is absolute/no-symlink, outside repository/output/registry/bundle,
  immutable mode `0o600`, at most `1_048_576` bytes, and bound into the runtime
  contract before discovery.
  For review purposes `heavy_execution_performed=false` means exactly: no
  production Module-A worker operation or capability, no read/open/map of any
  approved source video or checkpoint, no MPS/GPU/device or external model
  service access, and no embedding row/file publication. A CPU-only
  `weights=None` class/factory construction inside a test is permitted and is
  not represented as a zero model-initialization claim; it remains bounded by
  wall/CPU/RSS/process limits. This narrower definition is explicit and
  mechanically observable. The runner applies the policy to the complete
  process tree, records every relevant spawn/path/device/network/write event,
  and refuses unknown descendants or events. Reviewed Python cannot replace or
  suppress the out-of-process observation. Every review/runtime consumer
  reopens this policy file no-follow and verifies its bytes, provider build/
  signature and exact protected inputs; comparing an arbitrary 64-hex string is
  insufficient.
- `ReviewCheckResourceLimits` is exactly
  `{wall_time_nanoseconds=900000000000,cpu_time_nanoseconds=900000000000,peak_rss_bytes=2147483648,maximum_processes=32,maximum_open_fds=256,maximum_raw_output_bytes=8388608,maximum_attestation_bytes=8388607,maximum_final_output_bytes=16777216}`.
  `DiscoveryResourceLimits` is separately exactly
  `{wall_time_nanoseconds=900000000000,cpu_time_nanoseconds=900000000000,peak_rss_bytes=2147483648,maximum_processes=32,maximum_open_fds=256,maximum_audit_bytes=16777216,maximum_temporary_bytes=67108864}`.
  `ReviewProcessResourceObservation` is exactly
  `{ended_wall_time_nanoseconds,ended_cpu_time_nanoseconds,peak_rss_bytes,peak_processes,peak_open_fds,production_worker_spawns=0,approved_media_or_checkpoint_opens=0,device_or_model_service_accesses=0,embedding_publications=0,heavy_execution_event_projection_sha256,limit_exceeded=false}`.
  Every measured field is a non-boolean nonnegative integer no greater than its
  matching limit. The trusted runner applies OS-enforced process-tree limits
  before exec and event-audits every fork/exec/exit/open/close so a short-lived
  child cannot evade process/FD accounting; at-most-100-millisecond sampling is
  supplemental for RSS/CPU and not the sole enforcement. The trusted OS audit
  above supplies the four zero counters and the compact-canonical relevant-
  event-array projection; self-reported Python counters are rejected.
  `RawCaptureReceipt` is exactly
  `{temporary_parent_device,temporary_parent_inode,relative_path,device,inode,size_bytes,file_sha256}`.
  It describes the one random mode-`0o600`, no-follow regular raw capture under
  the attested `outputs/` temporary prefix; its relative path is exactly
  `outputs/<check_name>.<launch-nonce>.raw`. `DurableCheckOutputReceipt` is
  exactly
  `{absolute_path,parent_device,parent_inode,device,inode,size_bytes,file_sha256}`
  and describes the reopened fixed final under the baseline-bound check-output
  directory. `CheckOutputPublicationObservation` is separately exactly
  `{raw_capture_receipt,durable_output_receipt,raw_output_bytes,execution_attestation_bytes,published_output_bytes}`. It is
  computed only after the execution-attestation object (whose process field is
  the already frozen `ReviewProcessResourceObservation`) is serialized, so no
  size field hashes itself. Raw bytes, compact attestation bytes, and final file
  `attestation + LF + raw output` are capped before any durable-output stage open;
  `published_output_bytes == execution_attestation_bytes + 1 +
  raw_output_bytes` and equals both the durable receipt and
  `ReviewedCheckReceipt.output_size_bytes`. The raw count/hash equals the exact
  suffix after the first LF in the durable final. The raw capture is historical
  evidence after a successful seal: it is unlinked only after final reopen,
  external durable receipt construction and temporary-parent fsync; later
  loaders recompute its size/hash from that suffix rather than attempting to
  reopen the deleted temporary inode.
  `DiscoveryResourceObservation` is exactly
  `{process_resource_observation,audit_bytes,temporary_bytes_peak}`; byte fields
  are non-boolean nonnegative integers bounded by `DiscoveryResourceLimits` and
  equal the complete trusted audit transcript and FD-196 tree measurements.
  Discovery has no check-output attestation/final-file equation.
  `FreshReviewGovernanceObservation` is exactly
  `{review_scope=external_governance_only,model_execution_scope=review_reasoning_only,local_module_a_worker_or_media_execution_performed=false}`.
  The independently receipted fresh reviewer is an external governance trust
  boundary, not a child of the local OS review sandbox; its use of a reasoning
  model is explicitly outside the local Module-A heavy-execution assertion.
  `FreshReviewResourceSummary` is exactly
  `{governance_observation,fresh_review_artifact_bytes}`. It exists only
  in the later implementation-review summary, after the fresh-review artifact
  has been externally frozen; the fresh-review artifact itself contains its
  exact governance observation and never embeds its own byte size. The size equals
  the external receipt and is at most `16_777_216` bytes.
  On any resource, provider, serialization or cap failure the runner kills/
  reaps the complete tree and emits no check or review receipt.
  `RuntimeBuildResourceLimits` is exactly
  `{wall_time_nanoseconds=3600000000000,cpu_time_nanoseconds=3600000000000,peak_rss_bytes=2147483648,maximum_processes=4,maximum_open_fds=256,maximum_builder_log_bytes=16777216,maximum_runtime_manifest_bytes=67108864,maximum_runtime_contract_bytes=67108864,maximum_runtime_image_bytes=4294967296,maximum_temporary_bytes=4563402752,minimum_free_bytes_after_publication=3758096384}`.
  `RuntimeBuildResourceObservation` is exactly
  `{process_resource_observation,heavy_execution_observation,builder_log_bytes,runtime_tree_bytes_pre_manifest,temporary_bytes_peak,free_bytes_before_manifest_publication}`.
  The process row has the type above; the heavy observation has the full policy/
  process/event schema and must identify the sole `runtime_builder` tree. Builder
  and process zero counters/projection must be equal field-for-field. Builder
  log bytes are externally counted and bounded; there is no check-output
  attestation/final-file equation. `runtime_tree_bytes_pre_manifest` is exactly
  the sum of `size_bytes` for regular members in
  `ordered_tree_member_receipts`; directories and symlinks contribute zero and
  the special manifest, standalone contract, builder log and temporary files
  are excluded. It is computed once from that exact array and never includes a
  serialization that embeds the value itself.
  The manifest parent and contract parent have the same filesystem device.
  Before either rename, free bytes on that exact device must be at
  least `minimum_free_bytes_after_publication + maximum_runtime_manifest_bytes +
  maximum_runtime_contract_bytes`. The serialized contract must not exceed that
  maximum and the serialized manifest must not exceed its maximum. After
  no-clobber manifest and contract publication,
  each file/parent fsync and locked reopen, the external issuer measures on that
  same device
  non-boolean `postpublication_free_bytes`, requires it at least the minimum,
  and only then externally freezes the small `RuntimeSnapshotReceipt`. This
  future observation is intentionally absent from the prepublication manifest
  and contract bytes. Runtime-tree construction is a byte-copy/re-hash
  operation only and must also satisfy the exact heavy-execution audit with
  four zero event counts. `DiscoveryResourceLimits` apply separately to each
  discovery/fixed-point process, `ReviewCheckResourceLimits` to each evidence
  check, and `RuntimeBuildResourceLimits` to the image build. The external fresh
  reviewer uses the governance observation above rather than a fictitious local
  process/output limit. Caller input cannot raise any limit.
- `StoredArtifactReceipt` and `FileReceipt` retain the parent's exact six- and
  three-field shapes and are used only for the parent/TASK-0257/prior-attempt
  graph.
- `StaticInputContractReceipt`, `GenerationMemberReceipt`,
  `NamedHistoryArtifactReceipt`, `HistorySubjectReceipt`, and
  `RunHistoryContractReceipt` use only their separately frozen shapes in this
  amendment. `RunHistoryContractReceipt` is exactly
  `{run_identity_receipt,head_receipt,marker_count}` with its first two fields
  `ArtifactFileReceipt` and a non-boolean integer count in `[0,10]`.
- `RegistryArtifactPathReceipt` is exactly
  `{registry_role,sequence_ordinal,filename,absolute_path,artifact_sha256,file_sha256}`.
  The complete array starts with `admission_claim` and `admission_completion`
  rows whose ordinal is null, followed by every `history_marker` row with exact
  contiguous non-boolean integer ordinal `1..N`. Filenames equal the fixed
  claim/completion basenames or the sequence-derived marker basename; absolute
  paths equal the externally authorized registry directory joined to that
  filename. Missing, duplicate, reordered, noncontiguous, extra, aliased or
  self-discovered rows reject.
- `BootstrapLauncherReceipt` is exactly
  `{protocol,source_sha256,runtime_snapshot_receipt}`. Protocol is
  `task0258-module-a-verified-python-bootstrap-v2`; source hash is lowercase
  and must equal the `file_sha256` of the exact reviewed
  `scripts/task0258_module_a_verified_bootstrap.py` code row. The runtime value
  is the small exact retained `RuntimeSnapshotReceipt`; its separately reopened
  contract must match the full retained runtime. No writable interpreter path,
  inline large contract, alternate path, or caller-selected runtime is valid.

In the implementation-approval artifact, `parent_spec_approval_receipt` and
`amendment_fresh_review_receipt` are `ArtifactFileReceipt`;
`approved_parent_file_receipts` is the exact three-element parent order of
`NamedFileReceipt`; and `approved_amendment_file_receipt` is one
`ReviewedFileReceipt`. `implementation_scope_baseline_receipt` is an
`ArtifactFileReceipt` whose target parses as the exact baseline schema. In the
implementation-review artifact,
`amendment_implementation_approval_receipt` is `ArtifactFileReceipt`, all four
ordered file lists contain `ReviewedFileReceipt`, the bootstrap field is
`BootstrapLauncherReceipt`, check rows use the exact
`ReviewedCheckReceipt` fields listed above, and each check-input set uses its
closed namespace shape below. Its baseline receipt and delta equal the
approval/current-tree replay. In the rerun authorization, both
approval/review fields are `ArtifactFileReceipt`, the
amendment/code/test/runtime/configuration, implementation-scope baseline/delta,
bootstrap, check-input, and check-receipt fields are
the exact values from review, and the static-input contract uses its three-field type. No
nested receipt may use a different shape merely because its hashes happen to
match. Its `allowed_operations` must equal the fixed array above and is not a
receipt or caller-selected extension point.

Every v2 attempt, resume, embedding, verification-attempt, candidate-gate,
failure, and result field named `authorization_receipts` is the exact object
`{parent_spec_approval,amendment_implementation_approval,amended_implementation_review,rerun_authorization}`;
each value is `ArtifactFileReceipt` and key set/order is canonical. A provider
slot has exactly `{provider,verification_state,receipt}`. When state is
`verified`, receipt is non-null and its role is fixed: the four authorization
providers, run admission, candidate bundle, and canonical JSON providers use
`ArtifactFileReceipt`; run history uses `RunHistoryContractReceipt`; static
inputs use `StaticInputContractReceipt`; parent/TASK/prior attempt providers use
their parent receipt type; candidate physical members use
`GenerationMemberReceipt`; JSONL uses `FileReceipt`. When state is `failed` or
`not_reached`, receipt is null. There is no generic mapping-valued receipt and
no self-selected provider order.

Admission is a separate, non-heavy phase.
First build an opaque `VerifiedCompleteNoWritePreflight` by replaying every
authorization, immutable input receipt, path/alias/ancestor constraint, resource
threshold, worst-case disk allowance, and output-parent identity without any
write. It holds a no-follow exclusive flock on the existing output parent and
revalidates immediately before publication.

Using only that capability, hold both the output-parent flock and a no-follow
exclusive flock on the consumption registry. First no-clobber publish
`<authorization-artifact-sha256>.claim.json`, schema
`agu.task0258-module-a-v2-run-consumption-claim.v1`, binding the authorization
internal/file receipts, run ID, output root, one-time nonce, `state=claimed`,
and internal SHA. Then atomically publish a new output root whose sole member is
`run_admission.json`, schema
`agu.task0258-module-a-v2-run-admission.v1`. The artifact binds rerun
authorization internal/file receipts, claim internal/file receipts, the exact
claim nonce, exact run ID and output root, and the exact static-input contract,
`maximum_run_count=1`, `admission_state=admitted`,
`module_b_authorized=false`, timestamp, and internal SHA. Stage under the same
parent, fsync the file and stage, rename no-clobber to the absent output root,
and fsync the parent. Finally, while both locks remain held, no-clobber publish
`<authorization-artifact-sha256>.completed.json`, schema
`agu.task0258-module-a-v2-run-consumption-completed.v1`, binding the claim
internal/file receipts, exact root/run/nonce, `consumption_count=1`,
`state=completed`, and the first-published physical CAS. That CAS has exactly
`root_identity={device,inode}` and
`admission_identity={device,inode,size_bytes,internal_sha256,file_sha256}`,
captured no-follow after the root rename and parent fsync. Seal the completion's
internal SHA and fsync the registry directory before releasing either lock.

The three schemas above are closed. Claim has exactly `schema_version`,
`module_id`, `authorization_receipt`, `run_id`,
`output_root_absolute_path`, `nonce`, `state=claimed`, `created_at_utc`, and
`artifact_sha256`. Admission has exactly `schema_version`, `module_id`,
`authorization_receipt`, `claim_receipt`, `nonce`, `run_id`,
`output_root_absolute_path`, `static_input_contract`, `maximum_run_count=1`,
`admission_state=admitted`, `module_b_authorized=false`, `created_at_utc`, and
`artifact_sha256`. Completion has exactly `schema_version`, `module_id`,
`authorization_receipt`, `claim_receipt`, `admission_receipt`, `nonce`,
`run_id`, `output_root_absolute_path`, `consumption_count=1`,
`state=completed`, `root_identity`, `admission_identity`, `created_at_utc`, and
`artifact_sha256`.

All three module IDs are `existing-45-temporal-retrospective`; path and run ID
must equal rerun authorization; nonce is exactly 64 lowercase hexadecimal
characters and equal across all three. Receipt fields are exact
`HistoryArtifactReceipt` values. `root_identity` is exactly `{device,inode}`;
`admission_identity` is exactly
`{device,inode,size_bytes,internal_sha256,file_sha256}`. Identity/size values are
nonnegative integers, hashes lowercase, timestamps UTC RFC3339 strings, and no
unknown field or JSON type coercion is accepted.

### Review sandbox and authorized pre-import bootstrap

The implementation-review environment and the eventual production rerun use
two different, non-convertible trust boundaries. They are not two modes of one
repository bootstrap.

Before a rerun authorization exists, a trusted external runner creates an
OS-enforced review sandbox. A mutable repository path is never the evidence
execution root. Discovery and evidence are two different, non-convertible
executions; only the latter can yield a `ReviewedCheckReceipt`.

The runner begins each check with an exact discovery-only launch. It reserves
FD 193 for a read-only kernel-audit channel whose writer is held only by the
trusted runner/OS, FD 194 for the no-follow original
repository-root directory, FD 195 for a read-only discovery-attestation pipe,
and FD 196 for the no-follow temporary-ancestor directory. On fixed-point
validation iterations only, FD 197 is the read-only candidate namespace-provider
root; it is closed for the initial original-repository discovery. The common
driver additionally consumes FD 202 request, executes the regular FD 203 source,
reads an independent attestation copy from FD 204 and emits its one consumption
receipt on FD 205 exactly as in evidence. Pytest discovery also receives the
externally authenticated synthetic-only `AF_UNIX/SOCK_STREAM` provider-control socket on FD 206;
Ruff discovery requires FD 206 closed. FD 195 remains exclusively for the
discovery binder. The compact-canonical discovery
attestation is exactly
`{protocol=task0258-review-discovery-fd-v1,check_name,execution_kind=namespace_discovery,child_pid,command_sha256,repository_root_absolute_path,repository_root_device,repository_root_inode,temporary_ancestor_absolute_path,temporary_ancestor_device,temporary_ancestor_inode,audit_sink_device,audit_sink_inode,candidate_namespace_root_absolute_path,candidate_namespace_root_device,candidate_namespace_root_inode,namespace_provider_manifest_receipt,runtime_snapshot_receipt,review_driver_request_receipt,review_bootstrap_source_file_cas,review_bootstrap_source_size_bytes,review_bootstrap_source_sha256,driver_attestation_fd=204,driver_consumption_receipt_fd=205,worker_audit_provider_control_fd,review_heavy_execution_policy_receipt,exact_argv,exact_environment,resource_limits,expected_runtime_read_receipts,seed_namespace_projection_sha256,iteration_ordinal,nonce}`.
Candidate namespace path/identity/manifest are all null on iteration one and all
non-null/equal FD 197 on later validation iterations. Audit sink identity equals
FD 193; temp identity equals FD 196. No descriptor serves two authorities.
`expected_runtime_read_receipts` is null initially and is the prior iteration's
lexical runtime-read observation on later discovery iterations.
Iteration is a non-boolean integer in `[1,3]`; nonce is 64 lowercase
hexadecimal. `resource_limits` is the exact `DiscoveryResourceLimits` and applies
to this entire diagnostic process tree.
Runtime/request/source/policy fields are non-null on every iteration and obey
the evidence receipt/CAS/equality rules. The request has
`review_execution_kind=discovery`; an evidence request/attestation, wrong check
or wrong iteration rejects. `worker_audit_provider_control_fd=206` only for
pytest discovery and is null for Ruff discovery.
A reviewed binder consumes FDs 193-196, conditional FD 197 and pytest-only FD
206 and may mint only a
process-private `VerifiedReviewDiscoveryContext`. That context lets pytest
fixtures and the synthetic no-model dispatcher execute the same diagnostic
branches needed for discovery, but it cannot write a check-output receipt,
mint an evidence context, call a production API, access a real model/device,
or be converted/serialized. Discovery diagnostics are discarded.

After each discovery child is reaped, the runner constructs an unsealed
`DiscoveryExecutionObservation` with exactly
`{launch_attestation,launch_attestation_sha256,review_driver_consumption_receipt,observed_runtime_read_receipts,review_heavy_execution_observation,resource_observation,ordered_namespace_query_observations,exit_code}`.
It verifies the same FD-205, runtime, heavy-execution and resource observations
as evidence before using any row in a fixed-point union;
`resource_observation` is `DiscoveryResourceObservation`, whose nested process
row has the common type and whose two byte fields cover the audit/temp tree.
This object is never a
check receipt or capability and is discarded after its exact rows are folded
into the next candidate manifest. Missing, swapped or stale request/source/
attestation/consumption channels invalidate the iteration rather than silently
narrowing discovery.

To prevent discovery from stopping before fixture/loader branches execute,
there is one closed discovery traversal API listed below. It accepts one of the
same six exact operation-input manifests as production, invokes the identical
canonical reader/parser/path validator and loader-DAG traversal, but replaces
every capability-mint/side-effect target with a no-op observation. Its
`DiscoveryOnlyObservation` is exactly
`{schema_version=agu.task0258-module-a-discovery-only-observation.v1,module_id,operation,input_manifest_artifact_sha256,terminal_loader_node,read_outcome}`.
Operation is one fixed authorized operation name; terminal node is its exact
last loader-DAG node; outcome is `traversal_completed` or
`expected_validation_failure`. `input_manifest_artifact_sha256` is only the
already supplied manifest's verified internal hash; it is not the observation's
own hash. The observation is an unsealed process-local diagnostic and has no
own `artifact_sha256`, audit/query hash, receipt, or capability payload;
the external runner alone seals the complete post-exit audit transcript.
Synthetic discovery scenarios may consume only this observation. Evidence and
production entrypoints reject it by exact type and process epoch. There is no
cast, common token base, serialization or promotion edge to an evidence or
production capability.

The exact operation-to-terminal-node map is:

| Operation | `terminal_loader_node` |
| --- | --- |
| `preflight_and_publish_admission` | `worker_media_inputs` |
| `recover_admission_completion` | `recovery_claim_and_admission` |
| `publish_candidate` | `producer_invocation_inputs` |
| `seal_candidate_receipt_bundle` | `candidate_directory` |
| `postpublication_verify` | `verification_embeddings` |
| `load_existing_terminal` | `selected_terminal_artifact` |

No target callable or side-effect node is reachable in this table.

Under that context, a repository/production-zero-write, no-network/no-device
sandbox audits the complete
process tree's successful and failed `open/openat`, `stat/lstat/fstat/fstatat`,
`access`, `readlink`, `getdents`, `exec`, `mmap` and xattr path resolutions.
The audit records exact metadata, directory membership, symlink target,
hardlink equivalence and negative errno, not merely file-open events. It covers
data fixtures, manifests, subprocess scripts, conftest/plugins, dynamic loads,
metadata-only branches and negative lookups. Discovery exit status is not
evidence. An unaudited child, audit overflow, unknown filesystem operation,
path escape or unstable observation fails the review.

The only discovery writes are beneath the FD 196 directory authority, to the
closed relative prefixes `audit/`, `synthetic/`, `outputs/`, and `tmp/`.
Directories are mode `0o700`, files `0o600`, per-scenario byte ceilings equal
the synthetic schemas, and the whole process tree inherits the same MAC policy.
FD 193 is the sole child-visible audit channel; repository code cannot write
or forge it. The trusted runner's persisted audit is under `audit/`. Every repository, retained
runtime, approved production output/registry/bundle/model/media path and every
other absolute/relative write is denied. Thus diagnostic atomic/fsync/rename
scenarios remain executable without granting project or production writes.

Every enumerated directory member is represented in its exact getdents result;
any later stat/open/readlink/access on that child is a separate exact query.
Therefore an `os.listdir()` branch cannot smuggle an unrepresented placeholder
or metadata result into the sparse provider.

The runner unions those observations with the mandatory static roots, opens
every positive member no-follow, requires stable two-pass metadata/content,
and constructs a new random mode-`0o500` sparse namespace beneath the isolated
temporary ancestor. Regular bytes, directory entry sets/modes, safe symlink
targets, hardlink relations, xattrs and negative assertions exactly reproduce
the `ReviewedNamespaceQueryReceipt` rows. It fsyncs every backing file and directory,
removes write permission and replays the complete manifest. The runner then
attempts the check against that namespace under the same kernel audit. Any new
or different resolution denies the attempt and produces no evidence; the new
observation may be unioned into a fresh discovery/snapshot iteration. Exactly
one stable fixed point must be reached within three iterations. The final
receipt-bearing evidence run is a separate subsequent process over the frozen
fixed-point namespace; discovery contexts, PIDs, nonces, outputs and audit
handles are invalid in it.

The evidence run uses this private namespace as cwd; the OS sandbox denies every
read of the original repository and every repository-relative path result not
represented by the exact manifest. It also denies every other user-writable
absolute path except attested temporary sinks; only the read-only runtime image
and OS sealed-system files are readable. Thus an unobserved fixture, directory
entry or negative lookup fails instead of escaping, while replace-and-restore
of original paths cannot change evidence bytes or metadata. Matching tests,
conftest, plugins, fixtures and subprocess code remain regular non-symlinks.
The fixed-point namespace and audit records are deleted only after immutable
check output publication.

The runner also resolves every requested executable symlink component with
no-follow directory walks, opens the final regular executable, and verifies
stable device/inode/size/file SHA. Repository evidence and runtime are two
different images: (A) the per-check disposable namespace provider/mount bound
by FD 197 and its manifest, and (B) the retained full static runtime image with
only the closed runtime root IDs above. The runner never copies repository
members into B or runtime members into A. It finalizes and mounts both read-only,
with write/device/suid execution denied and execution permitted only for the
verified runtime tools. Evidence cwd is A; Python/Ruff and libraries come only
from B. Launch attestation binds both roots/manifests independently. The
original tool symlink/target and original repository are denied by the OS
sandbox. A writable clone, bind to the original tree, or hash-close-reopen of
an original path is forbidden. If the host cannot provide and attest such an
OS-enforced read-only mount, the check fails.

`SourceResolvedExecutableReceipt` is exactly
`{requested_absolute_path,ordered_symlink_chain,final_absolute_path,source_device,source_inode,size_bytes,file_sha256}`;
each symlink row is exactly `{absolute_path,target_text}`.
`RuntimeExecutableFileCAS` is exactly
`{root_id,relative_path,device,inode,size_bytes,file_sha256}` and describes the
already finalized image member, never the source inode. `ExecutableCopyReceipt`
is exactly
`{source_executable_receipt,runtime_member_location,size_bytes,file_sha256}`;
its location is `{root_id,relative_path}`, its size/hash equal both the source
receipt and the corresponding regular `RuntimeTreeMemberReceipt`, and it carries
no claim that source and copy device/inode match. At launch, FD 200 must equal a
fresh `RuntimeExecutableFileCAS` for that exact member.
`RuntimeSourceFileCAS` has the identical six-field shape but is a distinct role
restricted to the contract's bootstrap-source member. At review launch FD 203,
and at production/worker launch FD 4, must equal its fresh device/inode/size/hash
and be the actual `/dev/fd` script descriptor; role copying between executable
and source CAS rejects. Installed libraries
retain the parent's semantic version/build-info tradeoff, while the check also
freezes their actual evidence-run bytes in the exact `RuntimeSnapshotContract`
and ordered `RuntimeTreeMemberReceipt` vocabulary above. The externally sealed
manifest covers the resolved interpreter, `pyvenv.cfg`, base stdlib and
extension/dylib closure, site-packages and inert `.pth` bytes, Ruff/Pytest entry
modules, every dynamically audited runtime-library read, and every subprocess
tool. It is not limited to members lexically beneath `.venv`. Every original
path is assigned exactly once by the five-row origin table; the retained image
recreates all rows under one content-addressed namespace. The projection is the canonical member
array hash and is independently recomputable from the manifest and mounted
tree. Each non-system tool is copied into the read-only image; an allowed OS
tool is no-follow receipted and OS-snapshot bound. The evidence sandbox denies
an unlisted executable or external package path. Executable/library bytes
therefore cannot be swapped between receipt and use.

Before discovery, the trusted runner statically constructs the complete runtime
image by the full-tree algorithm above; review actual-read observations must be
a subset and never expand or narrow it. Every discovery, fixed-point and
evidence process runs from that same retained image. Members not read by review
remain authorized production runtime members because they were independently
sealed by the static algorithm; no member outside the static tree is allowed.
The runner publishes the exact runtime manifest and standalone contract
no-clobber outside the project/output/registry/bundle paths, mounts the tree
read-only, and records only the small `RuntimeSnapshotReceipt` in every
executable launch attestation and in `BootstrapLauncherReceipt`. This retained
image is not deleted after review and must remain physically identical through
fresh review, user rerun authorization and the authorized run. Building it is
a bounded no-model review transaction governed by the exact build limits and
observation in its manifest; exceeding disk/time/CPU/RSS/process/FD/output
limits or the post-publication reserve emits no runtime receipt. It grants no
execution authority.

For pytest/Ruff/format the runner directly executes the exact `-P -S` review
launcher and FD-202 request above; the immutable bootstrap performs the sole
`runpy` dispatch and there is no zsh, path-backed venv entry, `site` startup, or
`.pth` execution. The target argument arrays remain exact and independently
attested. The direct child environment is the sorted closed object of
the six one-thread entries
`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
`BLIS_NUM_THREADS=1`, `VECLIB_MAXIMUM_THREADS=1`, and
`NUMEXPR_NUM_THREADS=1`, plus `PYTHONHASHSEED=0`,
`PYTHONDONTWRITEBYTECODE=1`, `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, `LC_ALL=C`,
`PWD` equal to the snapshot root, `PATH` equal to the closed read-only-mount
tool directories from the runtime snapshot, and `TMPDIR`, `HOME`, `XDG_CONFIG_HOME`,
`RUFF_CACHE_DIR` beneath the temporary ancestor. `OLDPWD`, `SHLVL`,
`_`, `LOGNAME`, `__CF_USER_TEXT_ENCODING`, `PYTEST_ADDOPTS`, `PYTHONPATH`,
`PYTEST_PLUGINS`, `PYTHONSTARTUP`, `PYTHONINSPECT`, user-site variables and
every unlisted variable are absent. Direct `execve`/fd-exec must preserve this
exact environment; no shell-generated variables are tolerated. Diff uses the
trusted runner's fixed whitespace algorithm above, no child process, no Git and
no environment-dependent policy.

For the first five checks the runner reserves FD 197 for the read-only snapshot
root, FD 198 for a read-only launch-attestation pipe and FD 199 for the
no-follow temporary-ancestor directory descriptor. For each executable check,
FD 200 is the already verified regular executable inside the retained runtime
image and FD 201 is that image's no-follow root directory; both are closed for
`diff_check`. The review driver additionally receives an independent copy of
the compact launch attestation on read-only FD 204 and writes exactly one
bounded compact-canonical consumption frame to FD 205. FD 204 is never FD 198;
the evidence binder remains the sole consumer of FD 198. Pytest review drivers
also receive the externally authenticated synthetic-only `AF_UNIX/SOCK_STREAM` provider-control
socket on FD 206; Ruff/format/diff require it closed. The runner drains and
verifies FD 205 before accepting target output or exit status. The driver
closes FD 205 before `runpy` dispatch, so target code cannot append or forge
that frame. The binder fstats and re-hashes FD 200, verifies it equals
`runtime_executable_file_cas`, verifies that CAS size/hash equals the paired
source-to-copy receipt without comparing their device/inode, and opens the fixed runtime manifest only
relative to FD 201. The compact-canonical
launch attestation has exactly `protocol`, `check_name`, `execution_kind`,
`child_pid`, `command_sha256`, `repository_root_absolute_path`,
`repository_root_device`, `repository_root_inode`,
`snapshot_root_absolute_path`, `snapshot_root_device`, `snapshot_root_inode`,
`temporary_ancestor_absolute_path`, `temporary_ancestor_device`,
`temporary_ancestor_inode`, `allowed_check_output_absolute_path`,
`allowed_check_output_parent_device`, `allowed_check_output_parent_inode`,
`source_resolved_executable_receipt`, `runtime_executable_file_cas`,
`executable_fd`, `runtime_root_fd`,
`runtime_snapshot_receipt`, `review_driver_request_receipt`,
`review_bootstrap_source_file_cas`, `review_bootstrap_source_size_bytes`,
`review_bootstrap_source_sha256`,
`driver_attestation_fd`, `driver_consumption_receipt_fd`,
`worker_audit_provider_control_fd`,
`review_heavy_execution_policy_receipt`, `exact_argv`, `exact_environment`,
`resource_limits`, `expected_runtime_read_receipts`,
`ordered_snapshot_query_receipts`, `namespace_provider_manifest_receipt`,
`namespace_provider_root_identity`,
`check_input_receipt_set_sha256`, and
`nonce`. Protocol is `task0258-review-snapshot-fd-v2`; nonce is 64 lowercase
hexadecimal. `allowed_check_output_absolute_path` equals the baseline-bound
check-output directory joined to the current fixed `<check_name>.out`; its safe
ancestors and the two exact parent physical-identity fields equal the baseline
before child launch, but no
stage or final file may yet exist. `execution_kind=immutable_snapshot_exec` for the first four checks with a non-null
source and copied-image executable receipts, a runtime-snapshot receipt,
`executable_fd=200`,
`runtime_root_fd=201`, `driver_attestation_fd=204`,
`driver_consumption_receipt_fd=205`, exact argv/environment and child PID.
`worker_audit_provider_control_fd=206` for pytest and is null for the other
three executable/diff roles; that socket can authorize only temp-bound
synthetic no-model audit sessions under the current review nonce.
The source receipt exactly equals
the separately reopened verified `RuntimeSnapshotContract` field
`python_executable_copy_receipt.source_executable_receipt`;
the copied CAS location/size/hash equals that copy receipt and its exact tree
member, while only the copied CAS device/inode is compared to FD 200.
`review_driver_request_receipt` is the exact `ArtifactFileReceipt` of the
FD-202 frame including its final LF. Source size/hash describe the one regular
read-only runtime member actually executed through `/dev/fd/203`; stdin is
`/dev/null` and carries no source duplicate. Before spawn the trusted runner
no-follow opens, fstats and hashes FD 203 against
`RuntimeSnapshotContract.bootstrap_source_copy_receipt`, sets its offset to
zero, preserves that same descriptor through exec, and the exact argv names no
other script. At its first instruction the driver reads FD 204 and lseeks/
re-hashes FD 203, then requires attestation request/source values to equal the
actual request and already executed immutable member. This after-start replay
does not claim to authenticate untrusted source after execution: pre-execution
authenticity follows from the runner-held same descriptor and exact `/dev/fd`
argv. Its FD-205 frame has schema
`agu.task0258-review-driver-consumption.v1` and exactly
`{schema_version,module_id,child_pid,review_execution_kind,check_name,launch_attestation_sha256,review_driver_request_receipt,review_bootstrap_source_file_cas,review_bootstrap_source_size_bytes,review_bootstrap_source_sha256,artifact_sha256}`.
PID/kind/check equal the launch. Request, source, attestation and channel swaps,
cross-phase/check reuse, or supplying matching bytes only to the later binder
reject. The runner records and replays this exact object in post-exit evidence.
`resource_limits` is exact `ReviewCheckResourceLimits` for all five checks.
The original executable was no-follow snapshotted from its verified FD; actual
exec is the attested read-only-mount copy. For `diff_check`,
`execution_kind=trusted_runner_fixed_whitespace`, both executable receipt/CAS
fields are null,
runtime snapshot is null, argv is empty, environment is null and child PID is
the runner PID; request/source/FD fields and runtime-read rows are null or empty
exactly as their types require. Its independently verified heavy-execution
policy receipt remains non-null. For each executable check,
`expected_runtime_read_receipts` is the lexical duplicate-free subset
of exact `RuntimeTreeMemberReceipt` values and every row must occur in the full
runtime snapshot that the completed fixed-point discovery observed for that
check. It is an allowlist frozen before the evidence process, not a claim about
future observations. Snapshot rows are exact `ReviewedNamespaceQueryReceipt` values, with
paths relative to the original repository, and their compact-canonical hash
equals the check-specific input projection in the implementation review.
`namespace_provider_root_identity` is exact `{device,inode}`.
`namespace_provider_manifest_receipt` is `ArtifactFileReceipt` for exact schema
`agu.task0258-review-namespace-provider-manifest.v1`, whose fields are exactly
`schema_version`, `module_id`, `query_projection_sha256`,
`ordered_backing_member_receipts`, and `artifact_sha256`. Backing rows are
lexical exact `{relative_path,size_bytes,file_sha256}` regular files. The
manifest is the fixed regular root member
`.task0258-namespace-provider-manifest.json`, opened no-follow relative to FD
197; no caller path or alternate basename is accepted. The
provider serves virtual metadata/query results from the query table and content
only from these immutable backing members; neither copied kernel inode values
nor an unreceipted placeholder is exposed.

The runner captures raw output only beneath the temporary ancestor. During the
evidence process the OS audit independently records runtime-image reads. After exit
it replays every namespace query, provider manifest/root and inherited authority,
requires unchanged
identity/bytes/metadata/negative assertions and no unlisted audited resolution,
then builds an execution attestation
with exactly
`{launch_attestation,launch_attestation_sha256,review_driver_consumption_receipt,observed_runtime_read_receipts,review_heavy_execution_observation,resource_observation,post_snapshot_receipt_set_sha256,exit_code,raw_capture_receipt,durable_output_parent_identity}`.
`raw_capture_receipt` is the exact `RawCaptureReceipt` already measured from the
still-open temporary raw file. `durable_output_parent_identity` is exactly
`{absolute_path,device,inode}` and byte-equals the pre-existing check-output
directory identity frozen in the implementation-scope baseline and the launch
attestation's allowed-output parent fields. The attestation
does not and cannot claim the not-yet-created final stage/file inode.
For executable checks `review_driver_consumption_receipt` is the exact verified
FD-205 object above. `review_heavy_execution_observation` is exactly
`{policy_receipt,audit_started_before_first_child=true,audit_ended_after_last_reap=true,audit_overflow=false,unobserved_descendant_count=0,ordered_process_rows,process_tree_projection_sha256,production_worker_spawns=0,approved_media_or_checkpoint_opens=0,device_or_model_service_accesses=0,embedding_publications=0,ordered_relevant_events=[],relevant_event_projection_sha256}`.
Each process row is exactly
`{ordinal,pid,parent_ordinal,role,start_time_nanoseconds,executable_file_cas,provider_code_signature,source_file_cas,exit_disposition,exit_code_or_signal}`:
ordinals are contiguous from one, PID/start are non-boolean nonnegative, parent
ordinal is null only for the externally launched `runtime_builder` or
`review_driver`, role is one policy role,
`runtime_builder` has null executable/source CAS plus the exact policy-provider
code signature; every other role has `RuntimeExecutableFileCAS` and null
provider signature. `source_file_cas` is `RuntimeSourceFileCAS` for Python
driver/synthetic worker or null for a fixed native tool. Exit disposition is `exit_code` or `signal`
and the final scalar matches it. A relevant-event row, when diagnosing a failed
run before refusing a receipt, has exactly
`{ordinal,process_ordinal,event_kind,subject_role,normalized_target,device,inode,flags,result}`
under the policy's event-specific nullability. A successful receipt always has
the empty event array whose exact compact-canonical hash is
`relevant_event_projection_sha256`; process projection hashes the complete
ordered process array by the policy formula.
It is emitted by the trusted out-of-process audit after the whole tree is
reaped; its policy receipt equals the launch and verified runtime contract, and
its relevant-event projection equals
`resource_observation.heavy_execution_event_projection_sha256`. For
`diff_check` only `review_driver_consumption_receipt` is null; its heavy
observation has an empty process array and exact empty process/event
projections under the same non-null policy. A missing/drifted audit, nonzero or unknown
relevant event, unobserved descendant, or Python-supplied counter prevents
attestation publication.
`resource_observation` is the exact `ReviewProcessResourceObservation`; a missing sample,
counter overflow, exceeded cap, surviving descendant, output truncation, or
inability to observe the whole process tree prevents attestation publication.
`observed_runtime_read_receipts` is lexical, duplicate-free, externally audited,
and must exactly equal `launch_attestation.expected_runtime_read_receipts`;
subset-only or superset-only equality is insufficient.
Only after compact serialization does the runner enforce attestation, raw and
final output caps. It writes that compact JSON as the first line, one LF, then
raw output to one
mode-`0o600` stage under the approved durable check-output parent, fsyncs,
renames no-clobber to the fixed final basename, fsyncs the parent and reopens
it. It requires the final suffix size/hash to equal the historical raw capture,
constructs the external `DurableCheckOutputReceipt`, and only then unlinks the
raw capture and fsyncs its temporary parent. It then constructs the exact
`CheckOutputPublicationObservation` from the already serialized attestation,
historical raw receipt and reopened published file; that observation is stored only
in the later implementation-review resource summary and is independently
recomputed by its loader.
`post_snapshot_receipt_set_sha256` is the compact-canonical hash of exactly
`{ordered_snapshot_query_receipts,namespace_provider_manifest_receipt,namespace_provider_root_identity,runtime_snapshot_receipt,runtime_read_receipts}`,
where `runtime_read_receipts` is the observed array after replay. The verifier
also constructs the same five-field projection from the launch attestation,
using `expected_runtime_read_receipts` as `runtime_read_receipts`; the two
objects and their hashes must be equal. It is not the
whole launch-attestation hash. Exit code must be
zero. `ReviewedCheckReceipt.sandbox_attestation_sha256`
hashes this execution attestation and its output receipt binds the complete
file, so command, runner, immutable cwd, executable FD and tested bytes are one
evidence unit.

Inside a pytest discovery run, the reviewed discovery binder first consumes
FDs 193/194/195/196, conditional FD 197 and pytest-only provider-control FD
206, verifies the exact discovery attestation/PID/command/repository
root/temp/audit identities and mints one
`VerifiedReviewDiscoveryContext`. It can derive only a
`VerifiedSyntheticDiscoveryTransactionContext`, whose dispatcher runs the same
fixed no-model diagnostic branches and OS subprocess primitives but returns
unreceipted observations solely to the audit sink. Neither type is accepted by
the evidence binder, evidence sealer, implementation-review artifact builder or
any production API. Both expire at process exit and are poisoned at fork.

Inside a separate pytest evidence run, the reviewed evidence binder reads FD
198 once to EOF,
checks the attested PID equals its own PID, compares command hash, snapshot cwd
and FD 197 identity, executable receipt/FD, argv, actual closed environment and
snapshot manifest, reopens/fstats FD 199, authenticates/retains provider-control
FD 206 for synthetic sessions, then closes the attestation FD while
retaining the directory authorities required by synthetic tests. It registers
one process-private exact `VerifiedImplementationReviewSandboxContext` in that
pytest process. The token itself is never inherited or transferred across a
fork/exec/pickle boundary; only the exact synthetic dispatcher below may spawn
its independently attested test children. Ruff, format and diff do not mint a
context. Missing/reused/writable/wrong-number FD, wrong PID/nonce,
snapshot/path drift, output mismatch or a second bind rejects.

The runner, attested namespace-query provider, its sandbox policy, retained
runtime image and the OS are the review TCB. Repository code validates the runner-bound handles only to mint a
diagnostic test context; it does not turn them into production authority. The
sandbox permits only the five canonical non-heavy checks, the fresh read-only
review protocol, their explicit immutable check-output sinks, and
test-framework paths below the bound temporary ancestor. At the OS boundary it
denies MPS/device initialization, real source-video and checkpoint opens,
network access, approved production output/registry/bundle paths, Module B, and
all writes outside those sinks. This context exists only so implementation can
be tested and reviewed before a rerun authorization exists. It is not an
authorization artifact, carries no `BootstrapLauncherReceipt`, and can never be
serialized, promoted, converted, or accepted by a production side-effect API.

The external runner may derive one process-private
`VerifiedSyntheticModuleATransactionContext` from the review sandbox. It binds
the temporary ancestor's no-follow `{device,inode}`, requires
`execution_scope=synthetic_test_only`,
`production_model_workers_forbidden=true`, and
`model_or_device_access_allowed=false`, and has an empty production-path set.
The synthetic dispatcher accepts only a closed `SyntheticModuleAScenario` enum:
`admission_publish`, `admission_claim_crash`, `admission_root_crash`,
`admission_completion_recovery`,
`recoverable_attempt_publish`, `terminal_failure_publish`,
`candidate_publish`, `candidate_receipt_bundle_publish`,
`candidate_receipt_bundle_write_crash`,
`candidate_receipt_bundle_fsync_crash`,
`candidate_receipt_bundle_rename_crash`,
`candidate_receipt_bundle_stage_residue_rejection`, `verified_result_publish`,
`postverification_failure_publish`, `foreign_stage_rejection`, and
`path_alias_rejection`, plus `producer_worker_completed`,
`producer_worker_recoverable`, `producer_worker_terminal_failure`,
`verification_worker_completed`, `verification_worker_terminal_failure`,
`worker_parent_liveness_eof`, `worker_result_overflow`,
`worker_observation_status_mismatch`, `worker_stage_replacement`, and
`worker_respawn_rejection`, plus `worker_wrong_secret`,
`worker_reused_secret`, `worker_wrong_request_hash`, `worker_wrong_role`,
`worker_wrong_callable`, `worker_wrong_claim_envelope`,
`worker_wrong_source_hash`, `worker_wrong_parent_pid`, and
`worker_wrong_fd_layout`, `worker_wrong_marker_fd`,
`worker_audit_prepare_eof`, `worker_audit_provider_eof`, `worker_audit_wrong_policy`,
`worker_audit_wrong_pid`, `worker_audit_permit_before_arm`,
`worker_audit_metadata_denied`, and `worker_audit_event_omission`. It returns only a diagnostic
`SyntheticModuleAObservation` and cannot mint any production approval,
authorization, history, admission, candidate, result, or failure capability.
Every scenario must call the exact same private canonical encoder, no-follow
path validator, CAS checker, no-clobber rename, fsync, history transition and
topology primitives used by production; duplicating those primitives in a test
helper is forbidden. All transaction paths must resolve beneath the bound
temporary ancestor. Ordinary artifact scenarios use in-memory fake workers.
Every worker-protocol scenario whose observation table has a non-null child
creates a real fresh subprocess with
the exact production `posix_spawn` file actions, source/request/marker/audit-gate/
claim-envelope descriptors, daemon liveness watchdog, bounded result drain, wait-status parser,
stage-CAS verifier and no-respawn history transition. Their one closed child
target is exactly
`scripts.task0258_module_a_verified_bootstrap:_run_synthetic_no_model_worker`.
The immutable reviewed bootstrap source row and snapshot hash must match before
dispatch; another module, same short name, test helper, import fallback or
caller-selected callable rejects. This target can emit only the scenario's
fixed tiny canonical fixtures beneath the synthetic stage, is denied every
model/device/source-video/checkpoint/production path, and returns no capability.
It uses a distinct `agu.task0258-module-a-synthetic-worker-request.v1` that a
production worker or production wrapper rejects. That request has exactly
`schema_version`, `module_id`, `scenario`, `review_execution_kind`, `parent_pid`, `child_nonce`,
`launch_secret`, `review_namespace_source_sha256`,
`review_launch_attestation_sha256`, `temporary_ancestor_absolute_path`,
`temporary_ancestor_identity`, `private_stage_absolute_path`,
`private_stage_identity`, `read_isolation_policy`, `liveness_fd=5`, `result_fd=6`,
`admitted_marker_fd=7`, `audit_start_gate_fd=8`, `claim_envelope_fd=9`,
`sandbox_attestation_fd=10`, `temporary_ancestor_fd=11`, and
`artifact_sha256`. Nonces/secrets are distinct
64-character lowercase hex strings; identities are exact `{device,inode}` and
all paths are beneath the attested temporary ancestor. Request and policy bytes
obey the production `8_388_608`/`2_097_152` caps, so the same full-request
prepare framing is exercised.

Before claim publication the synthetic parent completes the same
prepare/prepared primitive over FD 206. It then no-clobber publishes beneath
that ancestor a
one-shot `agu.task0258-module-a-synthetic-worker-launch-claim.v1` with exactly
`schema_version`, `module_id`, `scenario`, `review_execution_kind`, `worker_request_artifact_sha256`,
`parent_pid`, `child_nonce`, `launch_secret_sha256`,
`review_namespace_source_sha256`, `review_launch_attestation_sha256`,
`read_isolation_policy_artifact_sha256`,
`provider_process_instance_id`, `audit_prepared_artifact_sha256`,
`temporary_ancestor_identity`, `private_stage_identity`, `liveness_fd=5`,
`result_fd=6`, `admitted_marker_fd=7`, `audit_start_gate_fd=8`,
`claim_envelope_fd=9`, `sandbox_attestation_fd=10`,
`temporary_ancestor_fd=11`, `state=claimed_once`, and `artifact_sha256`.
Secret hash is SHA-256 of the 32 decoded bytes, never the 64 UTF-8 hex
characters. Its synthetic claim is no-follow opened on FD7 and its FD9 envelope has distinct schema
`agu.task0258-module-a-synthetic-worker-claim-envelope.v1` and exactly
`schema_version`, `module_id`, `scenario`, `review_execution_kind`, `claim_relative_path`,
`expected_claim_artifact_sha256`, `expected_claim_file_sha256`,
`worker_request_artifact_sha256`, `review_launch_attestation_sha256`,
`read_isolation_policy_artifact_sha256`,
`provider_process_instance_id`, `audit_prepared_artifact_sha256`,
`child_nonce`, and `artifact_sha256`. It is never accepted by a production
worker.

The synthetic parent uses its attested `AF_UNIX/SOCK_STREAM` FD 206 to run the same
prepare/prepared/child-started/gate/finalize/post-exit-attestation primitive as
production, with a distinct policy that permits only the temporary ancestor and
no-model target. Child FD 8 is the provider gate, not the parent control socket;
the latter is closed/replaced by file actions. Synthetic marker FD 7 and
envelope FD 9 exercise the same future-marker one-way binding. A synthetic
attestation can never satisfy a production run-identity/policy/role check.
Every shared audit frame sets `worker_role=synthetic_worker`; a production role
or absent role rejects before claim publication or child permit.
The embedded `SyntheticWorkerReadIsolationPolicy` is compact-canonical and
exactly
`{schema_version=agu.task0258-module-a-synthetic-worker-read-isolation-policy.v1,module_id,provider_receipt,review_execution_kind,scenario,review_launch_attestation_sha256,child_nonce,temporary_ancestor_identity,private_stage_identity,runtime_snapshot_contract_input,ordered_allowed_read_rows,ordered_denied_read_rows,read_event_projection_protocol=compact-canonical-ordered-worker-read-events-sha256-v1,maximum_read_event_rows=4096,maximum_read_event_bytes=2097152,maximum_policy_bytes=2097152,artifact_sha256}`.
It uses the production discriminated read-row/event vocabulary but its complete
allow set is, in order, the same manifest-bound retained-runtime row, exact
OS-signed runtime rows and required safe ancestors, then verified source FD 4,
the envelope-bound marker FD 7, the attested inherited-directory FD 11, the one
private-stage subtree and the exact synthetic fixture files for that scenario.
The runtime input has the production exact type/equality, and provider reopens
the same contract/manifest before `prepared`. Its ordered deny rows are
the approved repository, production output/registry/bundle, all media/
checkpoint paths and every other temporary-ancestor member. Policy
role/kind/nonce/path identities equal request, claim, envelope and launch
attestation; the policy artifact SHA is copied into claim/envelope and provider
prepare. Model/device/network endpoints remain denied by the enclosing exact
review-heavy policy and are not misrepresented as filesystem rows. Unknown,
extra, cross-mode or production-shaped rows reject. The
provider's returned synthetic attestation uses the same frame-hash chain and
event super-schema and is exactly
`{schema_version=agu.task0258-module-a-synthetic-worker-read-isolation-attestation.v1,module_id,policy_artifact_sha256,provider_receipt,review_execution_kind,scenario,review_launch_attestation_sha256,child_pid,ordered_observed_process_ids,provider_process_instance_id,child_nonce,prepare_artifact_sha256,prepared_artifact_sha256,child_started_artifact_sha256,permit_artifact_sha256,finalize_artifact_sha256,audit_started_before_spawn=true,audit_ended_after_child_exit=true,audit_overflow=false,ordered_observed_read_events,read_event_projection_sha256,denied_read_attempt_count=0,unknown_read_attempt_count=0,artifact_sha256}`.
Its process array is exactly `[child_pid]`; all equality/event/nullability rules
are the production rules with the exact synthetic review-kind/scenario/
launch-attestation tuple replacing run identity/worker role. It is captured in
the immutable check output and can never be loaded as a
`ProductionWorkerReadIsolationAttestation`.

`review_execution_kind` is exactly `discovery` or `evidence`. Validation uses
this one closed mapping, not string equality across differently named fields:

| `review_execution_kind` | Attestation protocol | Attestation `execution_kind` | Attestation FD |
| --- | --- | --- | --- |
| `discovery` | `task0258-review-discovery-fd-v1` | `namespace_discovery` | 195 |
| `evidence` | `task0258-review-snapshot-fd-v2` | `immutable_snapshot_exec` | 198 |

The bound context, request, claim and envelope must select the same row.

In all three synthetic objects,
`review_launch_attestation_sha256 = SHA256(compact-canonical launch_attestation bytes)`
for the already available FD 198 evidence launch attestation, or the distinct
FD 195 discovery launch attestation when executing a discovery-only diagnostic
scenario. The request, durable claim, envelope and inherited attestation bytes
must agree exactly before spawn/target entry. This field never means the future
post-exit execution attestation or a `ReviewedCheckReceipt` hash.
In the request and claim, `review_namespace_source_sha256` equals the final
evidence namespace projection for an evidence scenario and the current
discovery seed projection for a discovery-only scenario; role/schema mismatch
rejects before spawn.

The test child operation is exactly `internal_synthetic_worker`. It is accepted
only from the immutable reviewed snapshot source with the synthetic schema,
read-only FD10 containing the original externally verified launch attestation,
directory FD11 matching the temporary ancestor, marker FD7, provider gate FD8
and envelope FD9. Before its fixed no-model
callable runs, it verifies every claim/request/source/attestation/path/PID/FD
field against actual descriptors and `os.getppid()`. It cannot select a module
or callable. The production `internal_worker` dispatch rejects this schema and
the synthetic dispatch rejects every production authorization, registry,
source-video, checkpoint or output-root path. The OS sandbox inherited from the
trusted review runner remains active in the child.

`SyntheticModuleAObservation` is exact
`{schema_version,module_id,scenario,execution_mode,child_pid,child_process_epoch_sha256,claim_receipt,private_stage_identity,result_frame_sha256,wait_status,protocol_outcome,residue_paths,artifact_sha256}`.
Schema version is exactly
`agu.task0258-module-a-synthetic-observation.v1`; module ID is fixed.
`execution_mode` is exactly `in_memory_transaction` or `real_subprocess` as
fixed below. A non-null `claim_receipt` is exact `ArtifactFileReceipt`; a
non-null `private_stage_identity` is exact `{device,inode}`. Every non-null process/frame/internal/file hash is a
64-character lowercase SHA-256 string; PIDs/identities are non-boolean
nonnegative integers.
`wait_status` is null or exactly `{kind,exit_code,signal}`: `kind=exited` has a
non-boolean integer exit and null signal, while `kind=signaled` has null exit
and one closed signal name. The following table is exhaustive. `P` means a
non-null value of the field's exact type; `N` means null; `F` means one valid
bounded result-frame hash. `E0`, `E1`, `E75`, and `E130` mean exact
`{kind=exited,exit_code=0|1|75|130,signal=null}`; `KS` means
`{kind=signaled,exit_code=null,signal=SIGKILL}`. Every literal scenario in a
comma-separated cell independently has that row's exact values.

| Scenarios | Mode | claim | stage | child/epoch | frame | wait | `protocol_outcome` | `residue_paths` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `admission_publish`, `admission_completion_recovery`, `recoverable_attempt_publish`, `terminal_failure_publish`, `candidate_publish`, `candidate_receipt_bundle_publish`, `verified_result_publish`, `postverification_failure_publish` | `in_memory_transaction` | N | N | N/N | N | N | `accepted_transaction` | `[]` |
| `admission_claim_crash` | `in_memory_transaction` | N | N | N/N | N | N | `simulated_crash_after_claim` | `[]` |
| `admission_root_crash` | `in_memory_transaction` | N | N | N/N | N | N | `simulated_crash_after_root_publish` | `[]` |
| `candidate_receipt_bundle_write_crash`, `candidate_receipt_bundle_fsync_crash` | `in_memory_transaction` | N | P | N/N | N | N | `simulated_crash_with_stage` | `[B]` |
| `candidate_receipt_bundle_rename_crash` | `in_memory_transaction` | N | N | N/N | N | N | `simulated_crash_after_final_rename` | `[]` |
| `candidate_receipt_bundle_stage_residue_rejection` | `in_memory_transaction` | N | P | N/N | N | N | `rejected_stage_residue` | `[B]` |
| `foreign_stage_rejection` | `in_memory_transaction` | N | N | N/N | N | N | `rejected_foreign_stage` | `[X]` |
| `path_alias_rejection` | `in_memory_transaction` | N | N | N/N | N | N | `rejected_path_alias` | `[]` |
| `producer_worker_completed`, `verification_worker_completed` | `real_subprocess` | P | P | P/P | F | E0 | `accepted_completed` | `[]` |
| `producer_worker_recoverable` | `real_subprocess` | P | P | P/P | F | E130 | `accepted_recoverable` | `[]` |
| `producer_worker_terminal_failure`, `verification_worker_terminal_failure` | `real_subprocess` | P | P | P/P | F | E75 | `accepted_terminal_failure` | `[]` |
| `worker_parent_liveness_eof` | `real_subprocess` | P | P | P/P | N | E75 | `terminated_liveness` | `[]` |
| `worker_result_overflow` | `real_subprocess` | P | P | P/P | N | KS | `killed_overflow` | `[]` |
| `worker_observation_status_mismatch` | `real_subprocess` | P | P | P/P | F | E0 | `rejected_status_mismatch` | `[]` |
| `worker_stage_replacement` | `real_subprocess` | P | P | P/P | F | E0 | `rejected_stage_cas` | `[S]` |
| `worker_wrong_role`, `worker_wrong_claim_envelope`, `worker_wrong_source_hash`, `worker_wrong_parent_pid`, `worker_wrong_fd_layout` | `real_subprocess` | P | P | P/P | N | E1 | `rejected_before_target` | `[]` |
| `worker_wrong_marker_fd` | `real_subprocess` | P | P | P/P | N | E1 | `rejected_marker_fd` | `[]` |
| `worker_audit_prepare_eof` | `real_subprocess` | N | N | N/N | N | N | `rejected_before_claim` | `[]` |
| `worker_audit_provider_eof`, `worker_audit_wrong_policy`, `worker_audit_wrong_pid`, `worker_audit_metadata_denied`, `worker_audit_event_omission` | `real_subprocess` | P | P | P/P | F | E0 | `rejected_audit_protocol` | `[]` |
| `worker_audit_permit_before_arm` | `real_subprocess` | P | P | N/N | N | N | `rejected_before_spawn` | `[]` |
| `worker_wrong_secret`, `worker_reused_secret`, `worker_wrong_request_hash`, `worker_wrong_callable`, `worker_respawn_rejection` | `real_subprocess` | P | P | N/N | N | N | `rejected_before_spawn` | `[]` |

The synthetic bundle fixture fixes authorization artifact SHA to 64 lowercase
`a` characters and final basename to `candidate-receipt-bundle.json`; `B`
expands to the one relative stage basename
`.aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.candidate-receipt-bundle.json.task0258-bundle-stage`.
`X` is exactly `foreign.task0258-stage`; `S` is exactly
`worker-private-stage`. `residue_paths` records only untrusted staging paths
remaining after dispatcher cleanup, never stable published targets or history
claims, and is the lexical array shown after token expansion. No other nullability,
outcome, wait status, signal, residue, or execution mode is legal.
The five candidate-bundle scenarios have a cumulative synthetic-write ceiling
of 2,097,152 bytes; every other in-memory scenario has 1,048,576 bytes. Every
real-subprocess scenario has a 1,048,576-byte private-stage ceiling plus the
separate total result-frame ceiling of 65,536 bytes including its LF. The
dispatcher counts actual written bytes and rejects before the next write would
cross either applicable cap; these constants are not caller inputs.

No production receipt or capability appears. The synthetic dispatcher
returns only diagnostic PID, FD/status and residue observations and cannot mint
a production fresh embedding. The production wrappers reject this context,
while tests assert by primitive-call identity that the synthetic and production
dispatchers share the same spawn, pipe, watchdog, drain, parser and CAS
implementation.
Mutation scenarios that invalidate the closed request or one-shot claim before
spawn must prove the shared spawn primitive was never called; mutations of
actual PID/FD/source/envelope/status after admission run a real child and prove
it exits before the no-model target counter increments.
`admission_claim_crash` stops after durable claim publication;
`admission_root_crash` stops after root rename and parent fsync but before the
completion marker; `admission_completion_recovery` starts from exactly that
state and exercises the sole recovery transition. The five bundle scenarios
exercise the same bundle-parent flock, per-filesystem reserve, stage write,
file fsync, no-clobber rename, parent fsync and residue rejection primitive as
the external production sealer.
The worker scenarios mechanically cover different child PID/epoch, fixed FD
layout, every duplicated claim field, launch-secret and request binding, normal EOF/reap, recoverable and
terminal mappings, parent-liveness EOF, a 65,537th result byte, mismatched
observation/wait status, stage replacement and a second spawn from one admitted
head. No scenario may import or call a model function.

Only after the implementation review is sealed and the user separately issues
the exact rerun authorization may the trusted external orchestration boundary
execute the reviewed source at
`scripts/task0258_module_a_verified_bootstrap.py` with the exact retained
read-only runtime image in `BootstrapLauncherReceipt`. The child never opens
that repository path. The runner first locks and replays the external runtime
manifest, every `RuntimeTreeMemberReceipt`, root physical identity and read-only
mount flags. It requires the image's `python_snapshot_location` to be a
regular executable, opens it no-follow relative to the retained root descriptor,
and keeps that descriptor open through spawn. This copied interpreter has its
image-local base stdlib/dylib closure and uses the manifest-built package path;
it has no active adjacent `pyvenv.cfg`, never runs `site`, and is not the
original `.venv/bin/python` symlink or writable uv/base target. The runner
replays the reviewed source row and
`RuntimeSnapshotContract.bootstrap_source_copy_receipt`, opens that copied
regular source no-follow relative to the retained root as FD 4, verifies a
fresh `RuntimeSourceFileCAS`, and sets its offset to zero. The source copy's
size/hash equal both the reviewed code row and
`BootstrapLauncherReceipt.source_sha256`; the child never opens the mutable
repository path. The runner binds stdin to read-only `/dev/null`, creates only
the request/control pipes needed by the operation, and spawns the verified
regular interpreter inside the unchanged read-only mount with exact argv
`[PYTHON, "-P", "-S", "/dev/fd/4", "--task0258-operation", OPERATION,
"--task0258-request-fd", "3", "--task0258-source-fd", "4"]`, an empty
repository import path, and only the
parent-frozen environment allowlist. `-P` suppresses unsafe path prepending and
`-S` suppresses automatic `site`; `-I`/`-E` are forbidden because
`PYTHONHASHSEED=0` must be consumed during interpreter initialization. Each
bootstrap and both independent workers attest `sys.flags.ignore_environment=0`,
`sys.flags.hash_randomization=0`, and the same fixed hashes of the literal test
strings `task0258` and `module-a-v2` before any model/device initialization.
The runner writes already-snapshotted compact-canonical request bytes to
inherited descriptor 3 and closes that write end. The interpreter executes the
same already-open immutable regular FD 4 verified by the runner; no source
bytes travel through stdin or a second source pipe. At its first instruction
the bootstrap lseeks and re-hashes FD 4 against the authorized launcher and
retains that descriptor/verified snapshot only for later role-specific child
bootstrap launches. This after-start replay is diagnostic; pre-execution
authenticity follows from the same open file description, exact `/dev/fd/4`
argv, retained read-only mount and trusted runner. Python never resolves or
reopens the repository bootstrap pathname. The runner replays the runtime-root descriptor,
interpreter FD, complete manifest and mount flags immediately before spawn and
after child exit. All producer/verification child spawns reuse this exact
runtime-root authority and interpreter; no worker resolves a writable Python
path. Any non-`/dev/null` stdin, nonzero source offset, descriptor substitution,
pre/post-stat drift, wrong source size/hash, runtime member/mount drift,
unexpected argv/environment entry, duplicate source channel, or unsupported
descriptor inheritance fails before any unverified repository byte executes.
For `publish_candidate` only, the runner also passes the independently
authenticated provider-control `AF_UNIX/SOCK_STREAM` as supervisor FD 8, bound to the
same provider receipt used by both embedded worker policies. Every other public
operation requires FD 8 closed. The bootstrap verifies socket type, connected
provider peer code signature and no pending frame before target dispatch; a
plain pipe/file/socketpair peer or caller-created JSON channel rejects.
The OS, external runner, pipe/descriptor adapter, content-addressed read-only
runtime image and externally authorized `BootstrapLauncherReceipt`
are the production pre-import TCB; no other repository CLI or module is an
authorized process entry by itself.

The request bytes have schema `agu.task0258-module-a-v2-bootstrap-request.v1`
and exactly `schema_version`, `module_id`, `operation`,
`rerun_authorization_absolute_path`,
`expected_rerun_authorization_artifact_sha256`,
`expected_rerun_authorization_file_sha256`,
`operation_input_manifest_absolute_path`,
`expected_operation_input_manifest_artifact_sha256`,
`expected_operation_input_manifest_file_sha256`, and `artifact_sha256`.
Module ID is fixed, operation is one exact member of the authorization's
`allowed_operations`, paths are absolute/no-symlink, hashes are lowercase, and
the request's internal SHA verifies compact canonical bytes. Descriptor 3
contains no module, callable, arbitrary argv, or Python expression.

Operation-input manifests are six different closed schemas, not one generic
mapping. They share only these exact nested types:

- `BootstrapArtifactInput` is exactly
  `{absolute_path,expected_artifact_sha256,expected_file_sha256}`.
- `BootstrapFileInput` is exactly
  `{absolute_path,expected_size_bytes,expected_file_sha256}`.
- `BootstrapPathMap` is an array of exact
  `{role,absolute_path,expected_size_bytes,expected_file_sha256}` rows in the
  role order fixed by the referenced approval or review; it cannot be empty
  when that referenced list is nonempty. For parent-spec maps, `role` is the
  exact receipt filename; for code/test/runtime/configuration maps it is the
  exact repository-relative receipt path. Their absolute path must be the
  approved repository root joined to that role and must match its receipt; no
  alternate slug or alias is accepted. Check-output roles are instead the six
  fixed check names and their absolute paths must equal the baseline-bound
  check-output directory joined to `<role>.out`, matching each
  `ReviewedCheckReceipt.output_path`. Neither join rule applies to check-input rows.
- `BootstrapNamespacePathMap` is an array of exact
  `{role,absolute_path,expected_query_receipt}` rows. It is used only for
  `approved_check_input_paths`. Role is exactly
  `<check-name>-<four-digit-query-ordinal>` and rows are ordered first by
  the five fixed checks and then by each check's lexical query rows. The
  expected value is that row's complete `ReviewedNamespaceQueryReceipt`, and
  `absolute_path` must equal the receipt-bound repository root joined to
  `expected_query_receipt.path`, never joined to the role slug. This permits
  one physical path to be independently bound to more than one check without a
  map collision while preserving exact path-to-receipt equality. Successful,
  metadata, directory and failed-query rows all participate; an error row
  validates safe ancestors and exact errno rather than requiring bytes.
- `AuthorizationReconstructionInputs` has exactly `parent_approval`,
  `parent_approved_spec_paths`, `expected_parent_fresh_review_internal_sha256`,
  `expected_parent_fresh_review_file_sha256`,
  `expected_parent_approval_statement_sha256`, `implementation_approval`,
  `amendment_path`, `amendment_review`, `implementation_scope_baseline`,
  `rerun_authorization`,
  `implementation_review`, `approved_code_paths`, `approved_test_paths`,
  `approved_runtime_dependency_paths`, `approved_check_configuration_paths`,
  `approved_check_input_paths`, `approved_check_output_paths`,
  `authorized_output_root_absolute_path`,
  `authorized_candidate_receipt_bundle_absolute_path`, and
  `authorized_registry_directory_absolute_path`.
  The approval/review/baseline artifact fields are `BootstrapArtifactInput`;
  amendment path is `BootstrapFileInput`; the
  six ordinary path collections are `BootstrapPathMap` and
  `approved_check_input_paths` is `BootstrapNamespacePathMap`; the three standalone expected
  hashes are lowercase SHA-256 strings. The final three fields are
  absolute/no-symlink paths and must equal the rerun-authorization bytes.
- `StaticReconstructionInputs` has exactly `temporal_plan`,
  `task0257_input_paths`, `expected_task0257_receipts`, and
  `expected_task0257_receipts_projection_sha256`. Temporal plan is
  `BootstrapArtifactInput`; the next two values are the complete exact parent
  dataclass projections with every path absolute/no-symlink and every parent
  receipt present in declared order; the projection hash is lowercase.
- `HistoryReconstructionInputs` is exactly
  `{registry_directory_absolute_path,expected_registry_receipts}`; the second
  field is the complete ordered `NamedHistoryArtifactReceipt` sequence.
- `AdmissionReconstructionInputs` is exactly `{admission}` where admission is
  `BootstrapArtifactInput`.
- `PriorAttemptReconstructionInputs` is exactly
  `{ordered_attempts,resume_absolute_path,expected_resume_cas}`.
  `ordered_attempts` is an increasing-ordinal array of exact
  `{attempt_directory_absolute_path,attempt_record_receipt,resource_log_receipt}`
  rows. Record uses parent `StoredArtifactReceipt` and log uses parent
  `FileReceipt`. Resume path and complete parent `ResumeCAS` are both null or
  both non-null and must describe the next resumable prefix; no directory
  discovery is permitted.
- `VerificationPriorAttemptHistoryInputs` is a distinct exact type
  `{ordered_attempts}`. Its increasing-ordinal rows are exactly
  `{attempt_record_absolute_path,attempt_record_receipt,resource_log_absolute_path,resource_log_receipt}`.
  The two paths identify only the externally receipted record and log regular
  files. No attempt-directory path, resume path/CAS, embedding path, feature
  matrix or directory descriptor occurs in this type. The verification child
  replays historical ResumeCAS metadata only as authenticated nested data in
  each verified attempt record; it has no authority to open a resume payload.
- `ProductionWorkerReadIsolationPolicy` is a compact-canonical embedded object
  with exactly
  `{schema_version=agu.task0258-module-a-worker-read-isolation-policy.v1,module_id,provider_receipt,run_identity_receipt,worker_role,child_nonce,output_root_identity,runtime_snapshot_contract_input,ordered_allowed_read_rows,ordered_denied_read_rows,read_event_projection_protocol,maximum_read_event_rows,maximum_read_event_bytes,maximum_policy_bytes=2097152,artifact_sha256}`.
  Provider is the same independently verified kernel process-audit provider as
  the review policy. `output_root_identity` is exactly
  `{absolute_path,device,inode}` and equals the held-lock run root;
  device/inode are non-boolean nonnegative integers. `read_event_projection_protocol` is exactly
  `compact-canonical-ordered-worker-read-events-sha256-v1`; event maxima are
  exactly `21600` and `16777216`. The final compact-canonical policy plus LF is
  at most `maximum_policy_bytes`; cap+1 rejects before marker publication.
  `runtime_snapshot_contract_input` is exactly
  `{runtime_root_absolute_path,runtime_root_device,runtime_root_inode,runtime_contract_absolute_path,runtime_contract_size_bytes,runtime_contract_receipt,runtime_manifest_absolute_path,runtime_manifest_size_bytes,runtime_manifest_receipt,runtime_tree_projection_sha256}`.
  Both receipts are exact `ArtifactFileReceipt`; sizes are non-boolean
  nonnegative integers. Before `prepared`, the provider independently no-follow
  opens/parses both absolute files, verifies sizes/internal/file hashes and the
  complete contract-to-manifest/root/projection equality, and builds its own
  immutable event matcher. Hash-only or supervisor-classified rows are not
  accepted. `ordered_allowed_read_rows` is a discriminated union.
  A path row is exactly
  `{locator_kind=path,path_role,absolute_path,fd_number=null,match_kind,entry_kind,expected_device,expected_inode,expected_size_bytes,expected_file_sha256,expected_symlink_target_text,expected_code_signature,ordered_allowed_operations}`;
  `match_kind` is `exact_regular`, `exact_directory`, `exact_symlink`,
  `exact_os_signed_regular`, or
  `owned_stage_subtree`, and conditional identity/
  size/hash values are all non-null for a regular file while a directory has
  size/hash/target null. A symlink has non-null identity and exact target text
  but null size/hash, and is accepted only when the sealed runtime contract
  names both that link and its resolved target; regular/directory target text is
  null. An OS-signed regular row has non-null identity/size/file hash and exact
  `{absolute_path,team_identifier,cdhash}` equal to one runtime-contract OS-TCB
  row; every other variant has `expected_code_signature=null`.
  `owned_stage_subtree` is allowed only for the already existing
  request-bound private stage directory: its root device/inode are non-null,
  size/hash are null, and its only future children are the role/disposition
  basenames in the staged-receipt table; each successful child-file event must
  later equal the worker observation receipt. The one verified bootstrap-source
  FD row is exactly
  `{locator_kind=verified_inherited_fd,path_role=bootstrap_source,absolute_path=null,fd_number=4,match_kind=exact_regular,entry_kind=regular,expected_device,expected_inode,expected_size_bytes,expected_file_sha256,ordered_allowed_operations=[fstat,open,read]}`;
  its four expected physical/byte values equal the request-authorized
  `RuntimeSourceFileCAS`, and an open of `/dev/fd/4` is normalized to this row
  rather than treated as general `/dev` access. The inherited-directory FD
  variant, used only by the synthetic sandbox, is exactly
  `{locator_kind=verified_inherited_directory_fd,path_role=temporary_ancestor,absolute_path=null,fd_number=11,match_kind=exact_directory,entry_kind=directory,expected_device,expected_inode,expected_size_bytes=null,expected_file_sha256=null,ordered_allowed_operations=[fstat,getdents]}`.
  Its identity equals the external review launch attestation and any production
  policy containing this variant rejects. The one manifest-bound runtime
  row is exactly
  `{locator_kind=manifest_bound_runtime_subtree,path_role=retained_runtime,absolute_path,runtime_root_device,runtime_root_inode,runtime_tree_projection_sha256,ordered_allowed_operations=[lstat,stat,access,readlink,open,fstat,getdents,read,mmap,exec]}`.
  Its root/projection equal `runtime_snapshot_contract_input`. Every actual
  descendant observation still uses `locator_kind=path`, an absolute normalized
  path and null FD; it matches this policy row only by the unique longest-prefix
  rule. A success must equal the complete manifest member kind/mode/identity/
  bytes/symlink target/directory children. An error is permitted only when the
  manifest proves that descendant absent, with exact `ENOENT` versus `ENOTDIR`
  recomputed from the longest existing prefix. Symlink escape, path outside the
  root, different errno, or a missing/extra manifest member rejects. Thus there
  is no special negative-query event locator. A future-marker row is exactly
  `{locator_kind=envelope_bound_marker_fd,path_role=admitted_marker,absolute_path=null,fd_number=7,match_kind=envelope_fresh_marker_receipt,entry_kind=regular,expected_device=null,expected_inode=null,expected_size_bytes=null,expected_file_sha256=null,ordered_allowed_operations=[fstat,read]}`.
  It is the sole pre-request wildcard for a future immutable input artifact:
  after marker publication its actual identity/size/hash must equal the
  transaction-local fresh marker receipt in the claim envelope before any
  project import/model/device access. No path open is permitted for that row.
  The owned-stage row is not an input wildcard: it authorizes only the
  request-bound worker's closed output basenames beneath the already
  identity-bound stage, and every produced regular file must acquire its exact
  receipt in the one child observation before the parent can accept any read.
  Each denied row is exactly
  `{path_role,absolute_path,match_kind}` where `match_kind` is exactly
  `exact_file`, `subtree`, or `remaining_output_root`.
  Verification path rows cover only the verified bootstrap/runtime/
  authorization/static/media/checkpoint inputs, exact `run_admission.json`,
  registry claim/completion/pre-existing history files, exact prior
  attempt-record/resource-log files, its own private verification stage and the
  registry directory solely for `getdents`. The envelope-bound marker FD is the
  final allowed row. Its denied rows, in precedence order,
  cover every prior `resume.json`, the complete producer private stage, every
  producer embedding path, the prospective candidate producer-embedding member,
  then every otherwise-unlisted member beneath the output root. Producer path
  rows cover the same verified bootstrap/runtime/authorization/static/media/
  checkpoint inputs, exact run admission, registry directory/claim/completion/
  pre-existing history, exact prior record/log files, the one current resume
  path and external `ResumeCAS` when non-null, its own stage, and the marker FD.
  Its denied rows cover every non-current resume, any pre-existing embedding or
  candidate member, every other worker stage, and the remaining output root.
  No other role-derived allow/deny row exists.
  Every absolute path authority also contributes its complete de-duplicated
  no-follow ancestor-directory chain. Those exact-directory rows are ordered by
  `(depth, UTF-8 absolute path)` and assigned
  `path_role=ancestor_directory_<four-digit-global-ordinal>`; their expected
  device/inode values come from the same under-lock path preflight and size/hash
  are null. No implicit or provider-discovered ancestor is accepted.
  An allowed `path_role` equals the originating provider/source ID or the exact
  registry basename role; stage/admission/marker roles use the literals in this
  section. Denied prior-resume roles are `prior_resume_<four-digit-ordinal>` in
  increasing ordinal, followed exactly by `producer_private_stage`, lexical
  `producer_embedding_<safe-role>` values, `candidate_producer_embedding`, and
  `remaining_output_root`; producer uses the analogous fixed applicable
  subset. An observed row copies its matched allow/deny `path_role`; an
  unmatched event uses `path_role=unknown` and therefore rejects.
  Rows are lexical within each fixed role group; path, casefold, ancestor,
  device/inode or hardlink ambiguity rejects. Reads under the output root are
  default-deny, not Python convention. Every root/path authority is completely
  derivable before request sealing; only owned-stage child bytes are completed
  by the child receipt and the explicitly typed marker-FD row is completed by
  the one-way envelope.
  Allowed rows are ordered in these exact groups: verified bootstrap-source FD;
  manifest-bound runtime subtree; de-duplicated safe ancestor directories;
  runtime-contract OS-TCB signature row order; authorization/static-input
  receipt order; checkpoint then four source videos;
  run admission; registry directory, claim, completion and pre-existing markers;
  prior attempt record/log order; owned stage; envelope-bound marker FD. Within
  a group lexical order applies. Exact regular JSON/data/log rows allow
  `[lstat,stat,access,open,fstat,read,mmap]`, executable runtime rows additionally allow
  `exec`, OS-signed runtime rows allow exactly
  `[lstat,stat,access,open,fstat,read,mmap,exec]`, sealed symlink rows allow only `[lstat,readlink]`, safe ancestor
  directories allow only `[lstat,access,open,fstat]`, final directory inputs
  allow `[lstat,stat,access,open,fstat,getdents]`, and the marker FD allows
  only `[fstat,read]`. The owned-stage-subtree row allows exactly
  `[lstat,stat,fstat,getdents,open,read,mmap]`; directory operations apply only
  to the bound root and regular-file operations only to the closed child
  basenames whose eventual staged receipts match. An operation not in the row
  is unknown and rejects.
- `ProductionWorkerReadIsolationAttestation` is the trusted provider's embedded
  compact-canonical object with exactly
  `{schema_version=agu.task0258-module-a-worker-read-isolation-attestation.v1,module_id,policy_artifact_sha256,provider_receipt,run_identity_receipt,worker_role,child_pid,ordered_observed_process_ids,provider_process_instance_id,child_nonce,prepare_artifact_sha256,prepared_artifact_sha256,child_started_artifact_sha256,permit_artifact_sha256,finalize_artifact_sha256,audit_started_before_spawn=true,audit_ended_after_child_exit=true,audit_overflow=false,ordered_observed_read_events,read_event_projection_sha256,denied_read_attempt_count=0,unknown_read_attempt_count=0,artifact_sha256}`.
  `provider_process_instance_id` is 64 lowercase hexadecimal characters issued
  by the external audit provider for this one prepared worker session. It is not
  the private capability-registry process epoch and is compared only with the
  provider prepare/permit/final frames below. `ordered_observed_process_ids` is
  exactly `[child_pid]`: the provider arms against the authenticated parent
  lineage before spawn and buffers every new-descendant namespace event from
  that boundary. After matching the child-started frame it assigns all buffered
  and subsequent events for that exact PID to the session before releasing the
  permit, excludes supervisor events, and rejects any unmatched/forked/spawned
  descendant or attribution gap rather than extending the array. Every event row has the one fixed
  super-schema
  `{ordinal,operation,path_role,locator_kind,normalized_path,fd_number,opened_fd_number,follow_symlinks,access_mode,access_granted,xattr_name,result_state,errno,entry_kind,mode_bits,device,inode,size_bytes,file_sha256,symlink_target_text,ordered_child_names,xattr_value_sha256}`.
  Ordinals are contiguous; operation is exactly `stat`, `lstat`, `fstat`,
  `access`, `readlink`, `getdents`, `xattr`, `read`, `open`, `mmap`, or `exec`.
  `result_state=error` has a nonzero integer errno and every returned-value field
  null. `result_state=success` has errno null: stat/lstat/fstat return exact
  kind/mode/device/inode/size; access has exact `access_mode` and boolean
  `access_granted`;
  readlink has target text; getdents has the lexical child array; xattr has name
  and value hash; `read`/open/mmap/exec of a regular file have physical identity,
  size/hash and their exact access mode. A non-null `entry_kind` is exactly
  `regular`, `directory`, `symlink`, or `other`; a successful matched event with
  `other` rejects. `access_granted` is null for every non-`access` operation,
  and all other inapplicable fields are null. For `access`, `access_mode` is a
  non-boolean integer from zero through seven using the POSIX `F_OK=0`,
  `X_OK=1`, `W_OK=2`, and `R_OK=4` bit values. For regular-file
  open/read/mmap/exec events it is exactly `read_only`, `read_shared`, or
  `execute` as applicable; all metadata/directory operations have it null. A
  `locator_kind=path` event has an absolute normalized path and null FD, while
  `locator_kind=verified_inherited_fd`, `verified_inherited_directory_fd`, or
  `envelope_bound_marker_fd` has a non-boolean nonnegative FD and null
  normalized path;
  any other combination rejects. Every successful event's role, locator,
  operation, physical identity and returned projection must match exactly one
  allowed row; every error event still matches the queried role/locator and
  exact errno. A successful `open` alone has non-null non-boolean
  `opened_fd_number`; every later fstat/read/mmap on that descriptor must link
  to the unique preceding open and role, while every non-open event has this
  field null. Reads of the fixed request/liveness/result/gate/envelope pipes or
  the provider-control socket are transport events, not filesystem namespace
  events and are excluded by exact descriptor type; a regular/directory/symlink
  descriptor can never use that exclusion. More than one matching policy row
  rejects as ambiguity.
  Follow-symlink policy is an exact boolean only for operations that accept it
  and null otherwise. Positive and negative observations are both recorded;
  denied-row matching takes precedence over allowed rows and metadata/directory
  operations cannot bypass it. Event count/serialized bytes are within the
  policy maxima. Provider, role, nonce and run identity equal the policy/request/
  actual process; a denied, unknown, missing, overflowed or unobserved namespace
  event makes the worker result unusable and no success capability is minted.
- `WorkerMediaInputs` is exactly
  `{checkpoint,ordered_source_videos,device,batch_size}`. Checkpoint is
  `BootstrapFileInput`; source rows are exact
  `{source_id,absolute_path,expected_size_bytes,expected_file_sha256}` in frozen
  game order; device equals the parent frozen value and batch size is integer
  one.

Each manifest has common exact fields `schema_version`, `module_id`,
`operation`, `authorization_inputs`, `static_inputs`, and `artifact_sha256`,
where authorization/static values use the types above. Its schema version and
additional exact fields are:

| Operation | Schema version | Additional exact fields |
| --- | --- | --- |
| `preflight_and_publish_admission` | `agu.task0258-module-a-v2-bootstrap-admission-inputs.v1` | `worker_media_inputs`, `output_root_absolute_path` |
| `recover_admission_completion` | `agu.task0258-module-a-v2-bootstrap-admission-recovery-inputs.v1` | `claim`, `admission`, `output_root_absolute_path` |
| `publish_candidate` | `agu.task0258-module-a-v2-bootstrap-candidate-inputs.v1` | `history_inputs`, `admission_inputs`, `prior_attempt_inputs`, `worker_media_inputs`, `output_root_absolute_path` |
| `seal_candidate_receipt_bundle` | `agu.task0258-module-a-v2-bootstrap-bundle-inputs.v1` | `history_inputs`, `admission_inputs`, `output_root_absolute_path`, `candidate_directory_absolute_path`, `candidate_receipt_bundle_absolute_path` |
| `postpublication_verify` | `agu.task0258-module-a-v2-bootstrap-postverify-inputs.v1` | `history_inputs`, `admission_inputs`, `prior_attempt_inputs`, `candidate_receipt_bundle`, `producer_embeddings`, `verification_embeddings`, `output_root_absolute_path` |
| `load_existing_terminal` | `agu.task0258-module-a-v2-bootstrap-terminal-inputs.v1` | `terminal_kind`, `history_inputs`, `admission_inputs`, `prior_attempt_inputs`, `candidate_receipt_bundle`, `terminal_artifact`, `terminal_member_receipts`, `producer_embeddings`, `verification_embeddings`, `output_root_absolute_path` |

`claim`, candidate bundle, terminal artifact and each embedding are
`BootstrapArtifactInput`. In the recovery manifest, `admission` is also that
type. `terminal_kind` is exactly `pre_candidate_failure`,
`postverification_failure`, or `verified_result`; fields not required by that
variant are null, not omitted: candidate bundle and both embeddings are null
for pre-candidate failure, while `terminal_member_receipts` is its exact ordered
`GenerationMemberReceipt` sequence; candidate bundle is non-null and both
embeddings plus terminal-member receipts are null for postverification failure;
candidate bundle and both embeddings are non-null while terminal-member
receipts are null for verified result. The wrapper rejects a null/non-null
pattern inconsistent with the selected stable topology.

The bootstrap reconstructs capabilities only through this loader DAG:
parent approval -> amendment implementation approval -> static inputs -> rerun
authorization -> run history -> run admission -> prior attempts ->
role-specific embeddings/candidate bundle -> selected target. Recovery stops
after completing parent approval -> implementation approval -> static inputs
-> rerun authorization, then skips the ordinary history/admission public
loaders and reconstructs the caller-receipted claim/admission only for the sole
recovery API. First admission likewise completes through rerun authorization
and static inputs, then stops before history/admission loaders. No loader may read a directory
to discover an input, derive its own expected receipt, consult process-global
paths, or accept an unknown manifest field. Missing, extra, aliased, reordered,
self-derived or operation-mismatched input fails before target dispatch.

After receipt-verifying the outer request and manifest bytes but before any
capability loader or other project input, the bootstrap requires the outer
`rerun_authorization_absolute_path` and both expected hashes to equal, field for
field, the nested `authorization_inputs.rerun_authorization`. Request operation
must equal manifest operation. The manifest's authorized output-root,
candidate-bundle and registry paths must equal both the outer authorization
bytes and every phase-specific duplicate path. No A/B authorization split,
path normalization difference, alternate run/root, or merely content-similar
receipt is accepted.

Internal worker requests are separately closed as
`agu.task0258-module-a-v2-worker-bootstrap-request.v1` with exactly
`schema_version`, `module_id`, `worker_role`, `parent_operation=publish_candidate`,
`authorization_inputs`, `static_inputs`, `history_inputs`, `admission_inputs`,
`prior_attempt_inputs`, `worker_media_inputs`, `read_isolation_policy`, `private_stage_absolute_path`,
`private_stage_identity`, `pre_claim_history_head_receipt`,
`admitted_marker_relative_path`, `parent_pid`, `child_nonce`, `launch_secret`,
`expected_hash_state_projection_sha256`,
`source_fd=4`, `liveness_fd=5`, `result_fd=6`, `admitted_marker_fd=7`,
`audit_start_gate_fd=8`, `claim_envelope_fd=9`, and
`artifact_sha256`.
The compact-canonical request plus final LF is at most `8_388_608` bytes and its
embedded policy is independently at most `2_097_152`; both exact lengths are
computed before provider prepare or marker publication. Prepare's complete
request plus canonical wrapper must fit its `16_777_216` stream-frame cap.
Role is exactly
`producer_worker` or `independent_empty_state_verification_worker`; nonce is 64
lowercase hexadecimal characters, and `launch_secret` is a different 64-character
lowercase hexadecimal value whose 32 decoded bytes, not its UTF-8 hex text, are
SHA-256 hashed to the value later stored in the
durable claim. Stage identity is exact `{device,inode}`.
`pre_claim_history_head_receipt` is the externally verified or private fresh
head that legally precedes this admitted event. The marker path is the unique
safe relative basename derived from the next sequence ordinal and role. It is
not a marker receipt and no final marker hash appears in the request.
`read_isolation_policy` is the exact role-derived
`ProductionWorkerReadIsolationPolicy`; its role, nonce, run identity and output
root equal the request and no caller supplies its row arrays. Producer
`prior_attempt_inputs` is `PriorAttemptReconstructionInputs`; verification uses
only `VerificationPriorAttemptHistoryInputs`. Any cross-role shape rejects
before bootstrap/model/device initialization.
`history_inputs` is a role-specific closed union. Producer uses
`{history_role=producer,ordered_registry_artifact_path_receipts,pre_claim_head_receipt}`
and the child reopens every claim/completion/marker row plus its subject
artifacts through the ordinary complete history/CAS loader. The array contains
exact `RegistryArtifactPathReceipt` values and byte-equals the complete
externally supplied registry sequence through the pre-claim head. Verification
instead uses
`{history_role=verification_predecessor,ordered_registry_artifact_path_receipts,pre_claim_head_receipt,producer_completed_private_marker_receipt,producer_subject_receipt_projection_sha256,producer_vector_access_forbidden=true}`.
Its array has the same exact row type and complete claim/completion/history
coverage. `producer_completed_private_marker_receipt` is an
`ArtifactFileReceipt` equal to the last row's two hashes, and the last row must
be the unique producer `completed_private` marker named by the pre-claim head.
`producer_subject_receipt_projection_sha256` is SHA-256 of compact canonical
JSON, without a final LF, of exactly
`{marker_artifact_sha256,subject_receipts,subject_projection_sha256,root_subject_cas}`
copied from that verified marker: `subject_receipts` is its exact ordered three
producer rows, `subject_projection_sha256` is the exact producer embedding
projection hash, and `root_subject_cas` is its exact ordered historical-stage
CAS array. Object keys sort canonically and array order is preserved. The child
recomputes this preimage only from verified marker metadata; another field set,
row order, nullable value, omitted claim/completion row or receipt-derived
shortcut rejects.
For that one role, `load_verification_predecessor_history` reopens and verifies
every canonical marker byte, artifact/file receipt, prior-head link, sequence,
run tuple and subject-receipt projection, but deliberately does not open,
map, hash or receive any path named by the producer completed-private
`root_subject_cas` or producer embedding receipt. It uses the role-specific
record/log paths above and never receives an attempt-directory descriptor.
The OS policy denies every stable prior resume, the producer private-stage
subtree, all producer embeddings and every unlisted output-root read. After reading the envelope, either child
requires the current registry listing to be exactly the frozen list followed
by the one receipted admitted marker; no directory discovery or extra entry is
tolerated.

Immediately before sealing the verification request, the parent fully reopens
and verifies the producer completed-private stage/CAS under both locks, mints a
process-private `FreshVerificationPredecessorHistory` containing the sanitized
projection above, and keeps all parent stage descriptors live. The explicit
child file-actions close those descriptors rather than inheriting them.
Immediately
after the child is reaped and before accepting any observation, the parent
performs the same full CAS/bytes replay again. Either mismatch fails the run;
the child audit must simultaneously produce the exact
`ProductionWorkerReadIsolationAttestation`; the attestation is verified before
any observation is accepted. The verification variant has no producer resume,
directory or vector field. The child repeats the same loader prefix in its
own PID/epoch and can reach only its role-specific worker callable. Its exact
argv uses the same verified regular `/dev/fd/4` source and FD-3 request layout with
`--task0258-operation internal_worker`; this literal is not a public
`allowed_operations` member and is accepted only with this worker schema,
`parent_operation=publish_candidate`, `parent_pid=os.getppid()`, the exact
pre-claim head, secret preimage, request artifact SHA and parent-created stage
CAS. The parent reopens the contract-bound bootstrap copy as a fresh
`RuntimeSourceFileCAS`, sets offset zero and passes only that descriptor as the
actual script; stdin remains `/dev/null`. The child also requires FD 5 to be a read-only liveness pipe from that
parent, FD 6 to be its only write-only result pipe, FD 7 to be the one read-only
regular admitted-marker descriptor, FD 8 to be the provider-owned read-only
audit-start gate, and FD 9 to be its only read-only claim-envelope pipe.

The dependency is deliberately one-way. The parent first seals the complete
request and obtains its internal artifact SHA, then completes the authenticated
provider `prepare`/`prepared` exchange below. It next publishes the admitted
history marker, which contains that request SHA and the prepared-session values
but whose receipt is not an input to the request. Only after the marker is
durable does the parent create a
compact-canonical transport envelope with schema
`agu.task0258-module-a-v2-worker-claim-envelope.v1` and exactly
`schema_version`, `module_id`, `admitted_marker_relative_path`,
`expected_marker_artifact_sha256`, `expected_marker_file_sha256`,
`worker_request_artifact_sha256`, `child_nonce`, `admitted_marker_fd=7`,
`provider_process_instance_id`, `audit_prepared_artifact_sha256`, and
`artifact_sha256`. After publishing the marker the parent opens it once
no-follow as a regular file, verifies identity/bytes against the new
transaction-local fresh-head receipt and passes that same open-file description
as child FD 7. The envelope
is sent once on FD 9 and is never stored as a receipt-bearing run artifact. The
child reads request and envelope once to EOF, fstats/hashes FD 7 without any
marker path open, verifies both internal hashes and the envelope-bound fresh
marker receipt, and requires marker request hash,
prior head, role, parent PID, nonce, provider process instance/prepared hash,
stage absolute/physical identity, secret
hash, verified bootstrap-source hash and all six worker FD numbers/modes to equal the
request and actual process. `os.getppid()` must equal both request and marker;
FD 5/8/9 are read-only pipes, FD 6 is a write-only pipe, FD 7 is the read-only
regular marker and the retained source FD 4 hashes to both the
request-authorized launcher and marker source hash.
This is the only route from the final marker receipt to the child and creates
no edge back into either hashed object.
An external/public request, wrong parent/secret/claim, absent/replaced stage,
reused admitted head, or attempt to dispatch a supervisor callable rejects.

The authorized supervisor alone inherits a full-duplex
`AF_UNIX/SOCK_STREAM` provider-control socket on parent FD 8 from the externally
attested production launcher; the descriptor is absent from every public API
and ordinary target. The bootstrap's peer-code-signature/socket-type/empty-
queue check occurs before private-stage creation and is the trust-spine provider
preflight. After that preflight, a later session transport failure is a
supervisor protocol failure rather than retroactive provider-identity drift: it
removes an owned pre-marker stage when still safe, or leaves an already
admitted head unresolved, but never publishes a result/failure. After sealing the complete request/stage CAS but before
publishing the admitted marker, it creates the audit-start pipe, sends
its write end to the provider with `SCM_RIGHTS`, and exchanges exact framed
records. Every record is `uint64_be(payload_length)` followed by exactly that
many compact-canonical JSON bytes and one final LF included in the length;
length zero, short read, cap+1, a missing/nonfinal LF, extra bytes before the
next state, ancillary data on a state that does not allow it, or anything other
than the one write-only gate-pipe FD on prepare rejects. All later records carry
no FD. `prepare` is
`{schema_version=agu.task0258-worker-audit-prepare.v1,module_id,worker_request,policy_artifact_sha256,worker_request_artifact_sha256,parent_pid,worker_role,child_nonce,artifact_sha256}`;
`worker_request` is the complete compact-canonical request object, not a hash or
caller summary. Provider reserializes it canonically, verifies its internal
artifact SHA and exact outer request hash, extracts/recomputes the embedded
read-isolation policy, and checks all duplicated role/parent/nonce values before
arming. Prepare is at most `16_777_216` bytes including LF; the other four
control records are each at most `4_096` bytes.
`prepared` is
`{schema_version=agu.task0258-worker-audit-prepared.v1,module_id,prepare_artifact_sha256,provider_process_instance_id,artifact_sha256}`.
Only after this valid frame does the supervisor publish the admitted marker;
its worker-launch claim and FD-9 envelope copy the provider process instance and
prepared artifact SHA exactly. If prepare/prepared fails, no marker is
published and the live supervisor removes/fsyncs its owned private stage before
returning; a crash residue still fails closed. If marker publication fails,
process exit/FD-8 EOF cancels the prepared provider session and no worker is
spawned. After a durable marker and `posix_spawn`, `child_started` is
`{schema_version=agu.task0258-worker-audit-child-started.v1,module_id,prepared_artifact_sha256,policy_artifact_sha256,worker_request_artifact_sha256,provider_process_instance_id,child_pid,parent_pid,worker_role,child_nonce,artifact_sha256}`.
After result-frame EOF plus `waitpid`, `finalize` is
`{schema_version=agu.task0258-worker-audit-finalize.v1,module_id,permit_artifact_sha256,policy_artifact_sha256,worker_request_artifact_sha256,provider_process_instance_id,child_pid,parent_pid,worker_role,exit_disposition,exit_code_or_signal,child_nonce,artifact_sha256}`.
`exit_disposition=exited` has a non-boolean integer
`exit_code_or_signal`; `exit_disposition=signaled` has one exact closed signal
name string. No other disposition/type is legal, and the value equals the
supervisor's raw `waitpid` interpretation and child observation rules below.
The provider arms the parent-descendant kernel audit before returning
`prepared`, matches the spawned PID/lineage, then writes exactly one `permit`
frame plus LF to the gate pipe. That frame is exactly
`{schema_version=agu.task0258-worker-audit-permit.v1,module_id,child_started_artifact_sha256,policy_artifact_sha256,worker_request_artifact_sha256,provider_process_instance_id,child_pid,parent_pid,worker_role,child_nonce,permit=true,artifact_sha256}`.
The provider immediately closes the gate writer after that one frame. The child
bootstrap reads exactly one permit frame to EOF from its FD 8
before any project import, input/model/device open or private-stage write.
The parent replaces/closes its provider-control FD in child file actions; the
child can neither read nor write the control socket. Prepared/child-started/
permit/finalize records are each at most 4,096 bytes; prepare has the distinct
complete-request cap above. After `finalize`, the
provider returns exactly one canonical
`ProductionWorkerReadIsolationAttestation` frame plus LF, at most 16,777,216
bytes as one length-framed record and ends that logical session. The authenticated
control socket remains open for the next legal worker in the same
`publish_candidate` operation: at most three producer sessions and, if and
only if a producer completes, exactly one subsequent verification session,
strictly sequential with fresh
`provider_process_instance_id` values and no interleaved frames. Before each
new prepare the socket must have no pending frame. The supervisor closes FD 8
when the operation reaches any terminal return; the provider then observes EOF
and accepts no further session for that process/run. Parent verifies provider code signature,
policy/request/PID/instance/nonce/status equality and requires the stream parser
to be exactly at a frame boundary with no buffered trailing byte before
constructing any final attempt artifact. Per-session EOF is neither expected
nor accepted while the operation-level socket remains reusable. Any byte later
arriving in a state other than the one expected by the closed session machine
rejects before another action. The five frame-artifact hashes embedded in the
attestation are exactly the prepare, prepared, child-started, permit and
finalize artifact hashes from this one chain; future verification
deterministically reconstructs those frame projections from the externally
receipted admitted marker plus the final attempt's embedded consumed request,
policy, observation, attestation and wait-status fields. It recomputes the full
prepare preimage and marker request artifact SHA; the raw transport stream is
not a stable artifact, while the identical post-consumption canonical request
object is. Any mismatch rejects. Missing, extra,
reordered, truncated, caller-made or
child-made frames, SCM_RIGHTS mismatch, permit-before-audit, provider EOF or cap
overflow is a supervisor protocol failure and cannot yield a completed attempt.

The internal dispatch table is exact:

| Worker role | Exact reviewed target |
| --- | --- |
| `producer_worker` | `scripts.extract_vru_causal_tiled_swin_embeddings:_run_authorized_producer_worker` |
| `independent_empty_state_verification_worker` | `scripts.extract_vru_causal_tiled_swin_embeddings:_run_authorized_empty_state_verification_worker` |

Neither callable is a public API. Their only return channel is FD 6 containing
one frame whose total length, including the one final LF, is at most 65,536
bytes. Its compact canonical JSON has schema
`agu.task0258-module-a-v2-worker-observation.v1` and exactly `schema_version`,
`module_id`, `worker_role`, `child_nonce`, `launch_claim_receipt`,
`private_stage_identity`, `disposition`, `exit_code`, `received_signal`,
`stop_reason`, `hash_state_projection_sha256`, `ordered_staged_receipts`,
`resource_counters`, `read_isolation_policy_artifact_sha256`, and
`artifact_sha256`.
`launch_claim_receipt` is exactly the envelope's admitted-marker
`HistoryArtifactReceipt`; child nonce and stage identity exactly equal request,
envelope and current no-follow stage CAS.
`read_isolation_policy_artifact_sha256` equals the exact embedded request policy
and the later trusted read-isolation attestation. The child cannot self-report
or omit the external audit; the parent accepts this frame only as one input to
the combined verification below.
`hash_state_projection_sha256` is SHA-256 of compact canonical
`{pythonhashseed_env="0",ignore_environment=0,hash_randomization=0,task0258_hash,module_a_v2_hash}`;
the last two fields are the non-boolean integer results of hashing the exact
UTF-8 literals named above. Parent and both worker observations must have the
same projection, and a second independent worker with another value is a
terminal environment failure.

Every staged-receipt row is exactly
`{provider,relative_path,receipt_kind,size_bytes,artifact_sha256,file_sha256}`.
Provider and path are the fixed values below; JSON uses `receipt_kind=json`
and two non-null lowercase hashes, while JSONL uses
`receipt_kind=file_only`, `artifact_sha256=null`, and a non-null file hash.
Size is a non-boolean nonnegative integer. Rows have exactly these orders:

| Role/disposition | Exact ordered providers and relative paths |
| --- | --- |
| producer / `completed` | `producer_resource_log:resource_guard.jsonl`, `producer_worker_payload:worker_payload.json` |
| producer / `interrupted_recoverable` | `producer_resource_log:resource_guard.jsonl`, `producer_worker_payload:worker_payload.json` |
| producer / `terminal_failure` | `producer_resource_log:resource_guard.jsonl` |
| verification / `completed` | `verification_resource_log:resource_guard.jsonl`, `verification_worker_payload:worker_payload.json` |
| verification / `terminal_failure` | `verification_resource_log:resource_guard.jsonl` |

`worker_payload.json` has exactly
`{schema_version=agu.task0258-module-a-worker-payload.v1,module_id,worker_role,child_nonce,run_identity_receipt,row_count,ordered_examples,computational_projection_sha256,artifact_sha256}`.
It contains only the raw role-specific rows produced in this child and no
attempt, embedding, resume or audit receipt. Row count/projection/examples obey
the same exact float32 projection as the later role-specific embedding; a
recoverable producer payload is the strictly advancing verified prefix. The
payload is transaction-local input to parent finalization, not a generation
member or public capability. Its compact-canonical bytes plus final LF are at
most `8_388_608`; cap+1 is a child protocol failure. Before writing the final
attempt, the parent replays the serialization-feasibility proof that already
ran before marker publication. For each possible role/disposition it builds the
exact canonical attempt skeleton with the five variable embedded values
(`worker_request`, `child_observation`, `worker_payload`,
`read_isolation_policy`, and `read_isolation_attestation`) set to null. The largest skeleton including all
other receipt arrays/keys/separators/final LF is at most `8_388_608`; the
conservative preclaim bound is therefore exactly
`8_388_608 + 8_388_608 + 65_536 + 8_388_608 + 2_097_152 + 16_777_216 = 44_105_728`, at most
the existing `67_108_864` attempt-record ceiling. The check uses the actual
policy/request lengths as well as these maxima, runs before provider prepare,
marker publication, model/device initialization or heavy work, and is repeated
before final write. Failure is a no-heavy no-marker refusal; the parent never
writes a partial oversized attempt.

`resource_counters` has exactly
`started_cumulative_active_runtime_nanoseconds`,
`ended_cumulative_active_runtime_nanoseconds`,
`started_cumulative_resource_samples`, `ended_cumulative_resource_samples`,
`started_cumulative_resource_log_bytes`, and
`ended_cumulative_resource_log_bytes`. Values are non-boolean nonnegative
integers and each ended value is not smaller than its start. For producer
observations the three starts equal the admitted head/prior attempt and the
three ends equal the independently measured child/log counters. Verification
uses the same exact six start/end values. Log size and final log row independently
recompute the ended byte/sample counters.

The observation/process mapping is closed. `completed` requires
`exit_code=0`, `received_signal=null`, `stop_reason=null`, a complete payload
and `waitpid` normal exit zero. Producer `interrupted_recoverable`
requires exactly (`SIGINT`,`external_sigint`,130) or
(`SIGTERM`,`external_sigterm`,143), a strictly advancing attempt 1 or 2 and a
valid resume; the verification role rejects this disposition. Terminal
`resource_breach`, `runtime_cap`, `sample_cap`, or `log_byte_cap` uses exit
`75`. Producer `SIGHUP` maps only to (`unsupported_signal`,129) and `SIGQUIT`
only to (`unsupported_signal`,131). A producer attempt-1/2 `SIGINT`/`SIGTERM`
without strict prefix progress maps only to (`non_advancing_prefix`,130/143),
and the same signals on attempt 3 before 45 rows map only to
(`attempt_limit`,130/143); a signal after row 45 follows completed finalization.
Verification `SIGINT` maps only to (`external_sigint`,130), `SIGTERM` only to
(`external_sigterm`,143), `SIGHUP` only to (`unsupported_signal`,129), and
`SIGQUIT` only to (`unsupported_signal`,131). Every other closed
parent/verification terminal stop reason
uses exit `1` and null signal. In every trusted terminal case the child
observation contains the identical role, disposition, signal, reason and
counters and `waitpid` reports the same normal exit. A signaled wait status, missing or
malformed observation, unexpected exit, status mismatch, EOF before one LF,
extra bytes, wrong claim/stage/role, or any unlisted combination is a supervisor
`worker_protocol_failure`, mints no fresh capability and is not reinterpreted as
a trusted child-declared failure. Because no exact trusted attempt observation
exists, a supervisor protocol failure has no publishable outer failure variant:
the admitted head remains unresolved, no failure artifact or resolution marker
is written, and every later invocation rejects without respawn or further heavy
work. The existing `schema_failure`, `input_identity_failure`, and
`unsupported_signal` trust-spine/attempt semantics remain unchanged; none is
synthesized from malformed transport.

After one valid child frame, EOF, matching `waitpid` and the independently
verified provider attestation, the parent—not the child—performs the sole final
artifact sealing in the still-private stage. It reopens the raw log/payload,
recomputes projection/counters, and embeds the exact child observation,
complete now-consumed worker request, complete worker payload plus the
read-isolation policy/attestation in the final attempt record. For completed or
recoverable producer work it seals the producer attempt first, then derives the
producer embedding or resume from the embedded payload so that the dependent
artifact can bind the attempt receipt. For completed verification it derives
the role-specific embedding from the payload first, then seals the verification
attempt that binds that embedding; a trusted terminal verification seals only
the attempt. It then reopens every final, proves exact payload-to-final
projection equality, unlinks `worker_payload.json`, fsyncs the stage and requires
the final stage member set to equal the existing attempt/log/embedding-or-resume
contract before appending a resolution marker. The child frame never names a
final attempt/embedding/resume receipt and the provider attestation never needs
to exist before child exit. Payload bytes remain durably replayable because the
complete canonical object is embedded in the attempt record; the transient
payload inode itself is not a future dependency. The request's launch secret is
revealed only inside this post-resolution attempt: its admitted head has already
been consumed and the monotone history forbids another spawn. Future verifiers
can therefore reconstruct the exact provider prepare frame without granting a
reusable launch capability.

The dispatch table is closed:

| Operation | Exact reviewed target |
| --- | --- |
| `preflight_and_publish_admission` | `scripts.extract_vru_causal_tiled_swin_embeddings:module_a_v2_preflight_and_publish_admission` |
| `recover_admission_completion` | `scripts.extract_vru_causal_tiled_swin_embeddings:recover_module_a_v2_run_admission_completion` |
| `publish_candidate` | `scripts.extract_vru_causal_tiled_swin_embeddings:publish_module_a_candidate` |
| `seal_candidate_receipt_bundle` | `scripts.screen_vru_causal_temporal_retrospective:seal_module_a_candidate_receipt_bundle` |
| `postpublication_verify` | `scripts.screen_vru_causal_temporal_retrospective:publish_module_a_postpublication_verification` |
| `load_existing_terminal` | `scripts.screen_vru_causal_temporal_retrospective:load_verified_existing_module_a_terminal` |

The temporal-plan sealer and analysis module are reviewed dependencies, never
production bootstrap targets. Except for the separately validated
`internal_worker` request above, the bootstrap rejects every operation outside
the public table, module, callable, request field or illegal state transition
before importing a target module.

The authorized bootstrap uses only frozen stdlib modules before verification.
It reads the rerun-authorization bytes once with no-follow,
duplicate-key/nonfinite rejection and caller-supplied external internal/file
hashes; requires the exact reviewed `BootstrapLauncherReceipt`; snapshots every
in-scope code and runtime-dependency file through no-follow descriptors; checks
path, device/inode, size and file SHA; recomputes the exact AST import closure;
and rejects drift before adding the repository to import resolution or opening
any project input/output. It never calls `site.main()`, `site.addsitedir`, or
executes/parses a `.pth` line: `.pth` members are sealed inert data only. It
constructs `sys.path` directly from the exact three-row Python path table and
manifest members, removes every path equal to/under the mutable repository and
every unlisted path/finder/hook, clears `sys.path_importer_cache`, and deletes
from `sys.modules` every protected top-level repository name and descendant.
It verifies the exact closed `sys.path`, `sys.meta_path`, `sys.path_hooks`,
`sys.path_importer_cache`, `builtins.__import__`, and allowed importlib-hook
projections. Only after that sanitation does it install the narrowly allowed
private meta-path loader that serves repository modules from immutable verified
byte snapshots. For the protected repository top-level
names derived from the reviewed closure (including `app`, `scripts`, and
`utils`), this finder is terminal: a name absent from the verified module map
raises `ModuleNotFoundError` and never falls through to another finder. The
bootstrap rechecks all import-state projections immediately before executing
the one exact target callable and requires every then-loaded protected module
to have the exact snapshot loader, frozen virtual filename, module-spec origin,
and source SHA from the reviewed module map. Any protected module preloaded by
stdlib, package initialization or retained runtime bytes rejects. No filesystem re-open or editable
finder may substitute executable local bytes. Before the first project input read, the loaded authorization code
repeats current-path receipt checks; a concurrent change after snapshot
therefore fails even though it cannot change the code being executed.

Successful production bootstrap creates a process-private, registry-backed
exact `VerifiedAuthorizedRerunBootstrapContext` containing the launcher and
interpreter CAS plus snapshot fingerprint. It cannot be serialized, copied,
constructed, subclassed, or minted by direct CLI invocation. The
rerun-authorization loader and every production side effect require this exact
subtype. Each production CLI rejects an absent, sandbox, synthetic, or otherwise
wrong context. Repository source-path replacement cannot affect the retained
immutable source FD/open file description actually executed, and runtime-member
mutation is caught by descriptor size/hash plus pre/post identity checks.
Direct `python script.py`, path-backed
bootstrap execution, or another argv/descriptor layout is explicitly
unsupported and fail-closed.

Every process-private context and capability registry binds the creating
`os.getpid()` plus a 32-byte `os.urandom` process epoch that is never exposed in
an artifact or public projection. Each token's hidden registry entry contains
that pair, and every lookup compares both before any path, input, lock or output
operation. Initialization installs `os.register_at_fork` hooks that, in the
child, poison and clear every Module-A capability registry, replace the process
epoch, close inherited Module-A lock/output descriptors, and make every
inherited token permanently invalid. `spawn`/pickle transfer is rejected by
the existing serialization guard. Producer and verification workers execute in
separate fresh child processes started only by the authorized parent supervisor.
Neither child inherits a parent capability, registry, flock, output descriptor,
model object, feature array, or Python module state. The parent uses
`posix_spawn` with explicit file actions and the same already-verified
bootstrap source copy; each child receives a fresh no-follow regular source FD
4 at offset zero, one role-specific compact-canonical request pipe, read-only
liveness FD 5, write-only result FD 6, read-only admitted-marker FD 7,
provider-owned audit-start gate FD 8 and read-only claim-envelope FD 9, creates its own
PID/epoch/context,
and independently replays authorization, its role-specific history view,
admission, static inputs, source-video/checkpoint receipts, environment and limits before
opening model inputs. The only internal child roles are `producer_worker` and
`independent_empty_state_verification_worker`; they are subordinate roles of
the authorized `publish_candidate` operation, not additional user-authorized
runs or public bootstrap operations.

Before spawning, the parent creates the private stage, captures its physical
CAS, generates the secret/nonce and exact request bytes, and completes the
provider prepare/prepared exchange. It then durably appends the admitted marker
containing the one-way request hash plus prepared-session identity while both
locks remain held, opens the marker as FD 7 and seals the nonpersistent claim
envelope for FD 9. Only after `posix_spawn` does it send the child-started frame;
the already armed provider then emits/closes the one permit gate. Its
private fresh-head token carries the secret and can be consumed exactly once by
the immediately following `posix_spawn`; no later invocation can reconstruct
it. The parent keeps the write end of FD 5 open until the child is reaped.

Before model/device initialization, the child starts one daemon watchdog that
performs only a blocking read on FD 5 and calls `os._exit(75)` on EOF/HUP or
malformed liveness data. On the normal path the worker main thread writes
exactly one bounded observation plus LF to FD 6, closes FD 6, and immediately
calls `os._exit` with the exact mapped status; it never returns through Python
thread shutdown. A daemon watchdog therefore cannot hold a successful process
open. The parent continuously drains FD 6. It accepts one complete frame only
after EOF, then calls `waitpid`, requires the closed observation/status mapping,
and only after reap closes the FD 5 writer. If a 65,537th byte arrives, the
parent sends `SIGKILL`, closes its pipe ends, reaps the child and records only a
supervisor protocol failure. The worker has no descendants.

If the authorized parent dies or its liveness writer closes unexpectedly, the
kernel closes the pipe and the daemon terminates the child with exit `75` at
its next scheduling opportunity. This is a fail-closed liveness boundary, not
an impossible guarantee that no instruction can retire between parent death
and watchdog scheduling. `parent_liveness_lost` never appears as a trusted
worker observation because no authorized parent remains to receive it; the
unresolved admitted marker permanently forbids respawn.

Each child may write only beneath the one parent-created, random, private stage
bound by CAS in its request and durable launch claim. It returns no token and
emits only its staged canonical resource log and optional raw worker payload
plus the closed receipt observation over FD 6. The parent reopens every staged
file no-follow, independently verifies exact bytes, receipts, role, run tuple,
resource counters and topology, waits/reaps, obtains the authenticated provider
attestation, builds and reopens the final attempt/dependent artifacts as above,
and only then mints the parent-PID
`FreshProducerTiledSwinEmbeddings` or
`FreshVerificationTiledSwinEmbeddings`. Child failure, inherited descriptor,
wrong PID/epoch/role/stage, or receipt drift mints no capability. The producer
and verification children use different process epochs and empty interpreter
state; the verification child receives neither producer vector nor resume
input. Direct fork inheritance remains poisoned as above.

Crash recovery is exact. Claim absent means never admitted. Claim present and
completion absent may finish the same nonce only when the already-published
root and exact admission exist; it captures their physical CAS and completes
the ledger. If the root is absent, the implementation cannot distinguish a
pre-rename crash from a published-then-moved root and therefore fails closed; it
must not republish admission bytes. Drift or any other root content also fails
closed. A completed marker always rejects another admission even if the output
root was renamed, deleted, or lost. The supervisor never deletes, overwrites,
repairs, or substitutes either marker.

The sole restart path for claim-present/completion-absent is
`recover_module_a_v2_run_admission_completion`. It is non-heavy and accepts the
authorized bootstrap/implementation/rerun/static capabilities plus
caller-frozen claim and admission internal/file receipts. Under both locks it
requires the registry to contain exactly the claim, the output root to contain
exactly the matching admission, replays authorization/static and all physical
CAS/path/disk checks, then publishes only the completion marker and returns
`None`. It cannot create or replace claim/admission, spawn a worker, or load a
complete history first. The caller freezes the new completion receipt before
the normal history loader can run.

The completed marker is the stable `run_identity_receipt` and the initial head
of an immutable run-history chain. Before any worker spawn, the supervisor holds
both locks, replays the complete registry and root, and no-clobber publishes the
corresponding admitted marker. Private completion markers are appended only
after their private subject bytes verify. A `*_published` marker is appended
only after its target has been renamed no-clobber, fsynced, re-opened and its
physical CAS captured, but before either lock is released. The canonical marker
schema is
`agu.task0258-module-a-v2-run-history-marker.v1`. Every marker has exactly
`schema_version`, `module_id`, `authorization_receipt`,
`run_identity_receipt`, `run_id`, `output_root`, `nonce`, `sequence_ordinal`,
`prior_marker_receipt`, `event`, `subject_kind`, `subject_receipts`,
`subject_projection_sha256`, `root_subject_cas`, `worker_launch_claim`,
`cumulative_active_runtime_nanoseconds`, `cumulative_resource_samples`,
`cumulative_resource_log_bytes`, `created_at_utc`, and `artifact_sha256`.
The first marker's prior receipt is the run identity; every later marker binds
the immediately preceding marker.

If a published-target marker cannot be appended or fsynced, the visible target
is untrusted, mints no capability, and may not be repaired, reused, or followed
by further work. It is not evidence that the history event occurred. A later
invocation sees target-without-marker and fails closed without writing.

For `subject_kind=worker_launch_admission`, receipts are an empty list,
projection is null, root CAS is a historical launch snapshot containing exactly the parent-created private stage
directory as the single seven-field row
`{relative_path,entry_kind=directory,device,inode,size_bytes=null,file_sha256=null,artifact_sha256=null}`,
and `worker_launch_claim` is non-null with
exactly `worker_role`, `parent_pid`, `child_nonce`,
`launch_secret_sha256`, `bootstrap_source_sha256`,
`worker_request_artifact_sha256`, `provider_process_instance_id`,
`audit_prepared_artifact_sha256`, `expected_hash_state_projection_sha256`,
`private_stage_identity`, `source_fd=4`, `liveness_fd=5`, `result_fd=6`,
`admitted_marker_fd=7`, `audit_start_gate_fd=8`, and
`claim_envelope_fd=9`.
`private_stage_identity` is exactly `{device,inode}` and equals only the
`device` and `inode` values of that directory CAS row; it is not a second CAS
shape.
hashes/nonces are lowercase and PID/FD/identity values are non-boolean
nonnegative integers. This shape is used only by producer/verification admitted
events and is re-opened only while verifying the immediately following worker
resolution edge. It is not a permanent stable-root member.
The launch claim is the unique deterministic projection of the sealed request
plus the already-verified bootstrap source: role, parent PID, nonce, decoded
secret hash, request artifact SHA, provider process instance/prepared SHA,
source SHA, stage identity and FD values are copied exactly; the parent's
startup hash-state projection is copied too, and no
caller supplies a second value. Parent and child independently
recompute the projection before spawn and before model/device initialization.
For every other subject kind, `worker_launch_claim=null`. For
`subject_kind=completed_private`, subject receipts name the just-produced
parent-finalized attempt/embedding artifacts, the transient worker payload is
already absent, the projection hash is non-null, and root
CAS is a historical resolution snapshot containing the same stage directory
followed by every staged file in lexical relative-path order. This does not
claim public path visibility and is re-opened only by the parent in the
uninterrupted resolution-to-verification-to-candidate/failure transaction. The
verification child sees only the sanitized predecessor-history projection and
has the complete provider-enforced producer-vector deny policy above. For
`subject_kind=published_paths`, subject receipts and exact no-follow root CAS
describe every newly visible directory/file, while projection is null. No other
nullability combination is valid.

The only event sequence is an ordered prefix of: producer attempt 1 admitted,
then either recoverable published, completed private, or terminal failure;
producer attempt 2/3 admitted and similarly resolved only when parent resume
permits; verification admitted then completed private or terminal failure;
candidate published; and finally verified result or postverification failure.
Event basenames are the zero-padded sequence ordinal plus the exact event name.
Before a worker starts, its admitted marker and one-shot launch claim must be
durable. Publishing that marker consumes the only launch allowance; a crash
before or during spawn is terminal and the marker can never authorize a second
child. A completed
producer/verification worker is followed by `completed_private`, not a false
publication claim. Recoverable attempts and terminal/candidate/result targets
use `published_paths`.

Every public invocation entry and future consumer requires a complete
externally receipted ordered marker list and exact registry coverage. While one
invocation continuously holds both locks, each append returns only a private
`FreshRunHistoryHead`; subsequent internal actions accept that fresh exact-type
token without pretending the external caller already froze it. On return or
resume, the fresh token expires and the external caller must freeze the current
marker set before another public invocation. An admitted marker lacking its
required resolution fails closed and may not respawn the worker. This includes
recoverable-looking bytes that are visible without their durable published
marker: they are untrusted crash residue and no later invocation may verify,
adopt, repair, or append a marker for them. Recoverable publication and its
marker therefore succeed only in one uninterrupted locked transaction.

A `completed_private` history-head capability authorizes only the exact next
edge in the transition graph. The fresh producer artifact capability itself
remains live through the same uninterrupted verification and
candidate/failure-sealing transaction; the fresh verification capability
likewise remains live through candidate/failure sealing. Neither may be used by
another run or unrelated transition. If the process returns, crashes, or loses
either lock before candidate/failure publication, both private capabilities are
unrecoverable and no public restart may recreate or consume them; the durable
marker makes the run fail closed without further heavy work.

Every action derives the legal next state from the chain rather than
caller-supplied prior paths. Ordinary deletion, replacement, or drift of
attempts, candidate, result, logs, plan, or another CAS-bound root member is
detected even when root/admission identities are unchanged. The sole exception
is a private-stage CAS already consumed by its exact resolution and later
candidate/failure publication edge: future stable loaders must require that
stage absent and verify the cryptographic resolution-to-published-member
mapping instead of reopening the historical path. An unresolved admitted or
completed-private head never gets this stable-path-deletion exception and
remains fail closed. The verification child's marker-only predecessor loader is
not such an exception: it cannot delete/consume the stage or mint a stable
history capability, while the parent performs the full pre/post CAS replay in
the same locked transaction. This local CAS does
not claim to distinguish an adversarial filesystem that reuses the same inode
and recreates byte-identical content; filesystem/registry rollback remains in
the trusted-authority tradeoff above. A terminal marker permits no later marker
or heavy action.

### Exact run-history registry encoding

Registry coverage is closed. The first two filenames are exactly
`<AUTH_SHA>.claim.json` and `<AUTH_SHA>.completed.json`, where `AUTH_SHA` is the
lowercase rerun-authorization internal SHA-256. History filenames are exactly
`<AUTH_SHA>.history-<NN>-<EVENT>.json`; `NN` starts at `01`, is exactly two
decimal digits, increments by one without gaps, and is at most `10`. `EVENT` is
exactly one of:

```text
producer_attempt_1_admitted
producer_attempt_1_recoverable_published
producer_attempt_1_completed_private
producer_attempt_2_admitted
producer_attempt_2_recoverable_published
producer_attempt_2_completed_private
producer_attempt_3_admitted
producer_attempt_3_completed_private
verification_attempt_admitted
verification_attempt_completed_private
pre_candidate_failure_published
candidate_published
verified_result_published
postverification_failure_published
```

The transition graph is exact: completed ledger goes only to producer attempt 1
admitted; an admitted producer attempt goes to its same-ordinal recoverable
published, completed private, or pre-candidate failure; a recoverable attempt N
goes only to attempt N+1 admitted;
a completed-private producer goes to verification admitted or pre-candidate
failure; verification admitted goes to verification completed private or
pre-candidate failure; verification completed private goes to candidate
published or pre-candidate failure; candidate published goes to verified result
or postverification failure. The three failure/result events are terminal.
Attempt 3 has no recoverable-published event because the parent attempt limit is
then exhausted.

Every run-history-only `HistoryArtifactReceipt` is exactly
`{artifact_sha256,file_sha256}` with lowercase SHA-256 strings. Every
`subject_receipts` uses the closed union `HistorySubjectReceipt`. A JSON row is
exactly `{provider,receipt_kind=json,artifact_sha256,file_sha256}` with two
lowercase hashes. A JSONL row is exactly
`{provider,receipt_kind=file_only,artifact_sha256=null,file_sha256}`. Provider
is a unique safe slug and rows are sorted by provider. No other nullability or
receipt kind is accepted.
`root_subject_cas` is an ordered list of rows with exactly `relative_path`,
`entry_kind`, `device`, `inode`, `size_bytes`, `file_sha256`, and
`artifact_sha256`. `entry_kind` is `directory`, `json_file`, or `jsonl_file`.
For a directory, size and both hashes are null; for JSON all are non-null; for
JSONL size/file hash are non-null and artifact hash is null. Integer identities
are nonnegative and paths are safe POSIX-relative strings sorted lexically.
`prior_marker_receipt` and `run_identity_receipt` use the exact history-receipt
shape. This new type does not replace, narrow, or alias the parent's six-field
`StoredArtifactReceipt`, which remains mandatory for prior producer attempts.
Conditional nulls for subject kind remain as defined above.

Marker scalars are exact. `schema_version` is
`agu.task0258-module-a-v2-run-history-marker.v1`; `module_id` is
`existing-45-temporal-retrospective`; authorization, identity, run ID, root and
nonce equal the stable run. `sequence_ordinal` is a non-boolean integer in
`[1,10]` equal to `NN` in the basename. Event equals the basename event and the
transition graph edge. All three cumulative fields are non-boolean,
nonnegative integers and never decrease. The first producer-attempt admission
sets all three to zero because its prior run-identity artifact has no counters;
every later admitted marker copies its prior history head. A producer resolution
or a pre-candidate failure containing a current producer attempt copies that
attempt's exact ended cumulative counters. Verification completion or a
pre-candidate failure containing a current verification attempt copies that
verification attempt's exact ended cumulative counters. Candidate, result,
postverification failure, and inter-phase/evaluator pre-candidate failure copy
their prior history head. Cumulative samples equal the last associated
resource-log row's `cumulative_sample_ordinal` when the current log is nonempty,
or the prior head when it is empty. Cumulative log bytes equal the actual byte
sum of every producer and verification JSONL file through that event: prior
published logs are bound by external receipts, while the current uninterrupted
transaction's private log is bound by its private verified file receipt and
canonical bytes. That sum equals the current attempt record's ended log-byte
counter; it does not wait for the external caller to freeze the current log
after return. Cumulative active runtime equals the attempt record's ended value;
when the log is nonempty it is not less than the last row's scheduled cumulative
time, but need not equal that sample time. Timestamps are
UTC RFC3339 strings; every non-null projection/internal/file value is lowercase
SHA-256. Boolean, float, stringified integer, NaN/Infinity, unknown field, or
coercion fails.

Marker bindings are event-exact. Every admitted event has
`subject_kind=worker_launch_admission`, an empty subject-receipt list, the one
historical private-stage directory CAS, a null projection, the non-null exact launch claim
and its prior history head. A producer `completed_private` event has exactly
`producer_resource_log`, `producer_attempt_record`, and `producer_embedding`
subject receipts plus the frozen embedding projection and the historical stage
directory/file CAS rows. Verification
`completed_private` similarly has exactly `verification_resource_log`,
`verification_attempt_record`, and `verification_embedding` plus its historical
stage CAS rows. A recoverable
published event has exactly its resource log, attempt record, and ResumeCAS
file in `subject_receipts`. Candidate published has exactly the ten candidate
file providers/paths in `subject_receipts`. Pre-candidate failure has exactly
the files frozen by its failed-phase table below. Verified result and
postverification failure each have exactly their one JSON file receipt. No
event may carry another provider receipt.

`root_subject_cas` is separate. For `published_paths` it describes every path made newly
visible by the one rename transaction: the published target directory first,
then every newly visible descendant directory and file in lexical
relative-path order. Candidate therefore has thirteen CAS rows: `candidate_v2`,
its two nested attempt directories, and its ten files. Result and
postverification failure each have two rows: the top-level directory and its
single JSON file. A pre-candidate failure has its top directory plus every
directory/file in the applicable table row. The first recoverable publication
atomically publishes `attempts` containing `attempt-0001` and records both
directories plus its three files; a later recoverable publication records the
new attempt directory plus its three files because `attempts` is already
stable. Subject receipt count is therefore not root-CAS count. Every CAS row is
cross-checked against the corresponding receipt when one exists.

For `worker_launch_admission` and `completed_private`, `root_subject_cas` is the
explicit historical-stage exception above, not a rename claim. The resolution
edge must match the admission directory identity; every resolution file CAS
must match its subject receipt. Candidate or pre-candidate-failure construction
then copies the verified bytes into its own private generation stage and checks
the exact old-to-new receipt projection. Consumed stage coverage is
phase-exact, never an unconditional pair:

| Publication or failed phase | Required consumed historical stage set |
| --- | --- |
| recoverable producer attempt | current producer stage only, projected to its attempt directory |
| producer worker or `pre_verification` failure | current or completed producer stage only |
| verification worker and every later pre-candidate phase | producer and verification stages |
| candidate | producer and verification stages |

An unlisted, missing or extra stage fails. For every listed stage the applicable
phase row binds its admitted/completed-private directory CAS, exact old member
receipts and exact destination generation-member receipts; it proves byte/hash
equality and then consumes that stage while holding both locks. After generation
publication it requires exactly those old paths and every other random stage
absent, and only then appends the published marker.
Future replay checks the completed-private receipts against the candidate or
failure member receipts and checks stage absence; it never requires a consumed
private-stage inode to remain. If cleanup or marker append fails, the target is
untrusted and no result capability can be minted.

Producer attempt/embedding v2 artifacts bind their same-ordinal admitted marker
as `history_head_receipt`; verification artifacts bind
`verification_attempt_admitted`; candidate gate binds
`verification_attempt_completed_private`; candidate receipt bundle and
postpublication verification bind `candidate_published`; the result registry
also binds `candidate_published`; and a public `VerifiedModuleAResult` requires
the externally receipted `verified_result_published` head. Each transition
verifier checks the named head and its strict-prefix ancestry back to the stable
run identity.

The public loader accepts one ordered `Sequence[NamedHistoryArtifactReceipt]`;
each row is exactly `{filename,artifact_sha256,file_sha256}`. It must equal the
complete actual registry listing in this exact order: claim, completion, then
history sequence. It includes no staging and permits no missing, duplicate,
extra, reordered, unknown, or self-discovered receipt. The loader independently
lists the locked directory, reads every named file once, verifies every byte and
chain edge, and rejects any other directory entry.

Registry publication uses exactly one transient basename,
`.<AUTH_SHA>.registry-stage`, opened with `O_CREAT|O_EXCL|O_NOFOLLOW`, mode
`0o600`, under both locks. A private transaction capability binds its
`(device,inode)`, intended final basename and exact bytes. It is the only entry
temporarily excluded from stable coverage while that same invocation publishes
one marker. After file fsync it is renamed no-clobber to the intended final
basename and the registry directory is fsynced. Stable loaders and future
invocations always require this stage absent. A crash residue is never deleted
or repaired and makes the run fail closed without further writes.

Creation returns only transaction-local private state. It authorizes no heavy
work. The external caller next freezes the admission, claim, and completion
internal/file receipts; a later invocation may load the public opaque
`VerifiedModuleAV2RunAdmission` and `VerifiedRunHistoryLedger`. All heavy
work requires both external receipt-bound tokens. The admission loader and
every later use compare the current root and admission file to the completion's
original physical CAS as well as all byte receipts; a byte-identical copy at a
new inode is rejected. A matching existing admission permits only resume of the
same physical run/root; any different or second admission rejects. A crash
before candidate publication does not restore the consumed allowance.

The
authorization loaders verify both internal and caller-frozen file SHA-256,
then bind the exact root identity before any input read, model/device access,
lock-file creation, directory creation, or output/log open. They retain safe
no-follow paths for every parent spec, the parent approval, amendment,
amendment review, implementation-scope baseline, amended code/test file,
complete runtime-dependency file, check-configuration file, implementation
review, and rerun
authorization named by their receipts. Every use re-opens all of those paths,
checks regular-file identity plus exact bytes and receipts, and rejects drift.
It independently re-walks the exact implementation-scope roots and requires the
approved ten-row delta plus byte-for-byte/metadata-for-metadata equality for
every non-authorized baseline entry; current dependency discovery cannot add
edit authority.
The rerun authorization's ordered code, test, runtime-dependency,
check-configuration, check-input, check-output, implementation-scope baseline
and delta values must equal the corresponding
ordered values in the implementation-review artifact. The
loader's safe code/test/runtime/configuration/check-input path maps and
check-output path map must have those exact keys and current namespace states;
each
check output is reopened no-follow and verified for size/file hash at every use.
For the first five it also parses the first attestation line and requires the
exact execution protocol, command SHA, check name, repository/snapshot identity,
resolved executable FD receipt, argv, closed direct-exec environment, complete
audited namespace equality (including metadata, directory and negative rows),
historical raw-capture receipt, durable-output parent identity and attestation SHA from its
`ReviewedCheckReceipt`; the execution receipt-set must equal the current
implementation-review check-input row and approved specification bytes.
It then replays the matching `CheckOutputPublicationObservation`, requires its
durable output physical/file receipt to equal the reopened final, and recomputes
the raw-capture size/hash from the final suffix; it never claims the deleted
temporary inode remains reopenable.
It also replays the standalone `RuntimeSnapshotContract`, retained manifest and
every member of the complete base-runtime/venv/site-package/bootstrap-source/
dylib/subprocess
tree against the attested `RuntimeSnapshotReceipt`. The retained read-only image is mandatory through the
authorized rerun; deletion or mount/member drift is a no-write refusal, not a
self-attested hash-only success.
The fresh-review row instead replays the sealed review artifact. No unattested
or stale inner-command log is accepted.
Reuse at another root, a second run, or Module B is rejected.
Candidate/failure publication and all future verification replay both
capabilities and the implementation-review receipt.

The consumption-registry owner is an explicit external monotone authority in
this threat model. It freezes the registry directory identity in the rerun
authorization and guarantees that claim, completion, and history children are
never deleted, replaced, rolled back, or hidden after publication. Every loader
also checks the frozen directory `(device,inode)`, exact caller-frozen registry
listing, and every child receipt. Accidental loss, replacement, partial drift,
or rollback relative to a supplied head is therefore a no-write refusal. A
malicious rollback by that same trusted authority cannot be detected by local
hashes and is not claimed to be prevented; a stronger boundary would require a
remote append-only/WORM ledger. `maximum_run_count=1` is enforced relative to
this explicitly trusted monotone authority, not against its compromise.

Parent capability objects never cross the amended public boundary directly.
`load_verified_module_a_static_inputs` accepts only the explicit parent plan
path with external internal/file receipts plus the complete parent
`Task0257InputPaths` and `Task0257ExpectedReceipts`. It internally invokes both
parent public verifiers, computes and checks the caller-frozen TASK-0257 receipt
projection, and copies only immutable canonical byte snapshots and receipt/path
fingerprints into a newly registered exact-type
`VerifiedModuleAStaticInputs`, and discards the temporary parent objects. The
wrapper exposes no mutable mapping, list, NumPy array, or parent token. Every
use re-opens all bound paths no-follow, replays both parent verifiers from the
independent external receipts, compares the complete immutable snapshot, and
rejects mutation, direct construction, `object.__new__`, token cloning,
subclassing, deserialization, or registry mismatch. Rerun authorization,
admission, producer, verification, candidate, failure, and result all bind the
same static-input contract; the rerun-authorization loader requires exact
equality with the wrapper's computed contract.

Whenever a persistent artifact needs to name this in-memory capability, it uses
the exact role-specific `StaticInputContractReceipt` shape
`{temporal_plan_artifact_sha256,temporal_plan_file_sha256,task0257_receipts_projection_sha256}`.
It equals the approved static-input contract byte-for-byte. It is not described
as a file receipt or capability receipt, carries no path, and never substitutes
for replaying the underlying plan and TASK-0257 inputs. Candidate, failure,
result, and postverification-failure provider slots use only this shape for the
single provider name `static_inputs`; they never invent an artifact SHA for the
opaque wrapper or duplicate plan/TASK-0257 logical providers.

The trust spine is global across admission, producer, verification, candidate,
failure, result, and every future use: parent/amendment/review/rerun
authorization; reviewed runtime closure; run identity and complete history;
admission/root physical CAS; plan/TASK-0257/checkpoint/source videos/prior chain;
authenticated kernel-audit provider/control peer and exact read policy; disk
reserve; exact target topology; and staging absence outside the one active
transaction. If any trust-spine check fails, the invocation is a receipt-free
no-write refusal: it returns nonzero, appends no marker, and publishes neither
terminal failure nor result. An already-admitted unresolved head then remains a
permanent fail-closed terminal dead state and can authorize no retry. This rule
overrides every later failure-publication clause. Durable failure is allowed
only for a downstream worker/computation/resource failure while the entire
trust spine still verifies immediately before publication.

This is an explicit, approval-visible supersession of the parent's blanket rule
that every post-admission receipt/schema/path/disk/atomic failure must yield a
stable `mechanical_failure`. That parent rule is preserved for an ordinary
downstream worker/computation/resource failure whose trust spine still verifies;
it is superseded only when authority, immutable input identity, reserve, target
topology, or safe publication itself is no longer trustworthy. Publishing a
failure from that state would assert provenance or atomicity the implementation
can no longer prove. Approval of this amendment's exact SHA approves this
conservative dead-state tradeoff; approval of the unchanged parent tuple alone
does not. The two authorization-bound external write transactions and the new
candidate/postpublication targets are the other explicit parent state-machine
changes. No evaluator receipt is added to a mechanical-failure generation.

Every amended v2 public producer entry point, including the tiled-Swin
supervisor CLI, requires these two verified authorization capabilities. No plan
sealer runs in the v2 output root: the already approved external plan is loaded
read-only only through `VerifiedModuleAStaticInputs`. Existing v1 plan/extractor
entry points explicitly reject a v2 output root, v2 schema, or v2 publication
target and cannot be used as an alternate path.

## Acyclic producer and verification artifacts

The producer extraction algorithm and recoverable resume semantics remain as
frozen in the parent specification, but every amended-run artifact is a v2
schema and is not interchangeable with an existing v1 artifact. Specifically,
producer attempt, resume, and embedding become
`agu.vru-causal-tiled-swin-attempt.v2`,
`agu.vru-causal-tiled-swin-embeddings-resume.v2`, and
`agu.vru-causal-tiled-swin-embeddings.v2`. Each exact v2 allowlist is its parent
v1 allowlist plus `authorization_receipts`, `run_identity_receipt`,
`history_head_receipt`, and `run_admission_receipt`. Producer attempt records,
but not resume or embedding artifacts, additionally contain exact
`worker_request`, `child_observation`, `worker_payload`, `read_isolation_policy`
and `read_isolation_attestation` objects of the five
closed types above; payload is non-null for completed/recoverable and null for a
trusted child-declared terminal disposition. Their provider/run/role/nonce equality is mandatory and the
attempt cannot be sealed as completed until the trusted audit has ended after
child exit and the parent has reaped it. Any denied/unknown/missing/malformed/
overflowed provider audit is a supervisor isolation-protocol failure: it seals
no attempt, embedding, resume or outer failure, leaves the admitted head
unresolved and permanently forbids respawn. Every publishable producer attempt,
including a trusted child-declared terminal failure, therefore has both policy
and successful non-null attestation.
`authorization_receipts` contains parent approval, amendment implementation
approval, amended implementation review, and exact rerun authorization
internal/file receipts. Every prior-attempt and ResumeCAS link must carry the
same exact authorization/run-identity/admission tuple. Their history-head
receipts may differ only as a legal strict-prefix progression in the one exact
chain. Any v1 schema, cross-run tuple, rollback, fork, or non-prefix head is
rejected.

The verification embedding and attempt schemas in this amendment are likewise
versioned v2 and include those same four receipt fields. Their public loaders require
the verified implementation approval, rerun authorization, and run admission
capabilities in addition to the plan, and replay every bound authorization path
at use. Candidate construction does not accept a caller-created public
embedding token: the supervisor's single candidate transaction passes
unexported exact-type `FreshProducerTiledSwinEmbeddings` and
`FreshVerificationTiledSwinEmbeddings` capabilities created by the two workers.
Those private capabilities carry the same run tuple and cannot be constructed,
copied, or used by public callers. Public receipt-bound embedding capabilities
exist only for the later post-publication replay.

After the producer reaches 45 verified rows, but before
candidate publication, the supervisor starts one independent verification
worker from an empty prefix. It uses a new process, fresh model construction,
and the same verified plan, four videos, Swin checkpoint, decoder, transform,
MPS device, batch size, environment, and resource limits. It may not read the
producer vectors, producer resume, or a caller feature matrix. This prohibition
is enforced and persisted by the verification attempt's role-specific
read-isolation policy/attestation, not by target-code convention.

The following hash dependency order is mandatory and acyclic:

```text
approved plan and TASK-0257 receipts
  + amendment implementation approval
  + implementation review and exact v2 rerun authorization
  -> claim -> run admission -> stable run identity
  -> producer request -> producer admitted head -> nonpersistent claim envelope
  -> private producer attempt/embedding
  -> producer completed-private head -> verification request
  -> verification admitted head -> nonpersistent claim envelope
  -> private verification attempt/embedding -> verification completed-private head
  -> candidate gate/member bytes -> candidate-published head
  -> caller-frozen candidate receipt bundle -> post-publication registry bytes
  -> result-published head -> caller-frozen registry and history receipts
```

### Verification embedding

`agu.vru-causal-tiled-swin-verification-embeddings.v2` is a role-specific
schema. Its exact fields are the common false eligibility fields plus:

```text
role = independent_empty_state_verification
authorization_receipts
run_identity_receipt
history_head_receipt
run_admission_receipt
plan_receipt
task0257_input_receipts
representation
producer_environment
checkpoint_receipt
source_video_receipts
row_count = 45
examples
artifact_sha256
```

Its example schema and float32 derivation are identical to the producer
embedding schema, but it has no producer `attempt_chain` and no receipt for its
own later verification-attempt record. Its loader returns only
`VerifiedVerificationTiledSwinEmbeddings`; the producer loader returns only
`VerifiedProducerTiledSwinEmbeddings`. The two capabilities must bind different
regular-file paths and different `(device,inode)` identities. Role substitution
or one artifact copied byte-for-byte into the other role is rejected.

### Verification attempt

After verification-attempt admission, always seal
`agu.vru-causal-tiled-swin-verification-attempt.v2` when the supervisor can
publish a trusted failure or success. Its exact fields are the common false
eligibility fields plus:

```text
verification_ordinal = 1
authorization_receipts
run_identity_receipt
history_head_receipt
run_admission_receipt
plan_receipt
task0257_input_receipts
checkpoint_receipt
source_video_receipts
producer_attempt_chain_receipts
producer_embedding_receipt
worker_request
child_observation
worker_payload
read_isolation_policy
read_isolation_attestation
verification_embedding_slot
resource_log_receipt
started_cumulative_active_runtime_nanoseconds
ended_cumulative_active_runtime_nanoseconds
started_cumulative_resource_samples
ended_cumulative_resource_samples
started_cumulative_resource_log_bytes
ended_cumulative_resource_log_bytes
computational_projection_sha256
received_signal
disposition
stop_reason
artifact_sha256
```

`verification_embedding_slot` is exactly
`{provider,verification_state,receipt}`. The provider is
`verification_tiled_swin_embeddings`; state is `verified`, `failed`, or
`not_reached`; only `verified` has a non-null receipt. A completed attempt
requires a verified slot, `received_signal=null`, `disposition=completed`, and
`stop_reason=null`. `child_observation` is the exact verified FD-6 object and
its role/nonce/disposition/status/counters/payload receipt equal the final
attempt, `waitpid` and embedded payload. `worker_request` is the complete
consumed request object whose artifact SHA equals the admitted marker, provider
prepare/frame chain and child observation; its role/history/policy/input values
equal the other attempt fields. `worker_payload` is the complete canonical child payload on
completed disposition and null for a child-declared terminal failure; its
projection/examples equal the verified embedding before attempt sealing.
`read_isolation_policy` and
`read_isolation_attestation` are the exact embedded objects from this worker
run. Their provider/run/role/PID/provider-process-instance/nonce/projection
equality, all-zero
denied/unknown counters, complete allowed-read event match and audit time bounds
are independently replayed by the attempt loader and every candidate/result
verifier. Any prior `resume.json`, producer private-stage/embedding, prospective
candidate producer-member, unlisted output-root, missing event or audit overflow
forces terminal failure and cannot yield a verified slot.
An isolation/provider failure follows the unresolved supervisor rule above and
does not create this attempt record; every publishable verification attempt has
both objects non-null, even when the trusted child disposition is terminal.
`computational_projection_sha256` is then the SHA-256 of
compact canonical JSON with the unique shape
`{"representation": REPRESENTATION_NAME, "ordered_examples": ROWS}`. `ROWS` is
the 45-element plan order; every row has exactly `ordinal`, `key`,
`tile_frame_indexes`, `derivation_only_tile_embeddings`, and `model_input`.
`key` has exactly `source_video_sha256`, `candidate_bundle_sha256`, and
`event_id`; `tile_frame_indexes` is the frozen 4-by-16 integer matrix;
`derivation_only_tile_embeddings` is the frozen 4-by-768 float32 projection;
and `model_input` is the frozen 1536-element float32 projection. This is the
embedding component of the parent computational projection in
`solution.md` and adds or omits no field. A failed attempt requires
`disposition=terminal_failure`; its projection hash is the same value only when
the embedding slot is verified and otherwise is null.

For failure, `received_signal` is null, `SIGINT`, `SIGTERM`, `SIGHUP`, or
`SIGQUIT`. `SIGINT` maps only to `external_sigint`, `SIGTERM` only to
`external_sigterm`, and `SIGHUP`/`SIGQUIT` only to `unsupported_signal`. With a
null signal, `stop_reason` is exactly one of `receipt_failure`,
`schema_failure`, `input_identity_failure`, `disk_failure`, `resource_breach`,
`runtime_cap`, `sample_cap`, `log_byte_cap`, `decode_failure`, `model_failure`,
`determinism_failure`, or `publication_failure`. No other signal/reason pairing
is valid. The embedding slot reflects whether embedding publication was
verified, failed, or not reached. The attempt therefore binds an already
sealed embedding only on success or a verified post-embedding failure; it never
creates an embedding-to-attempt edge, so no receipt cycle exists.

It has no resume and never increments the parent producer-attempt ordinal. Any
signal, guard/cap, receipt, input, decode, model, determinism, disk, or
publication failure is terminal. A failure before verification-attempt
admission has no verification attempt/log and leaves its provider slots
`not_reached`; a failure after admission must publish the failed verification
attempt and its complete log if trusted terminal publication remains possible.

### Candidate generation

`candidate_v2` is one no-clobber directory transaction with exactly:

```text
terminal_attempt/resource_guard.jsonl
terminal_attempt/attempt_record.json
producer_tiled_swin_embeddings.json
verification_attempt/resource_guard.jsonl
verification_attempt/attempt_record.json
verification_tiled_swin_embeddings.json
temporal_retrospective.json
baseline_final_evaluator.json
candidate_final_evaluator.json
candidate_gate.json
```

`agu.vru-causal-temporal-candidate-gate.v2` has the common false eligibility
fields plus exactly:

```text
input_receipts
producer_attempt_chain
verification_attempt_receipt
ordered_prepublication_check_results
error_bound_result
candidate_metric_outcome
publication_state = not_yet_observed
postpublication_verification_required = true
conditional_downstream
artifact_sha256
```

`candidate_metric_outcome` is only `within_frozen_error_bounds` or
`outside_frozen_error_bounds`. It is not `mechanical_pass`,
`temporal-hypothesis-rejected`, or `mechanical_failure`.
`conditional_downstream` sets every authority to false. The ordered checks end
at `candidate_publication_preconditions`; they contain no assertion that
candidate publication, residue absence, or post-publication verification has
already happened.

The candidate gate's input receipts have exact provider slots for parent spec
approval, amendment implementation approval, amended implementation review,
exact v2 rerun authorization, run history ledger, run admission, static inputs,
producer embedding, verification
embedding, verification attempt, retrospective, baseline evaluator, and
candidate evaluator. A receipt is present only when the corresponding provider
was independently verified.

Every check row in candidate, failure, and result artifacts has exactly
`{check_name,passed}` with a safe enum name and an exact boolean. Candidate
`ordered_prepublication_check_results` is this closed sequence, all true:

```text
parent_spec_authorization
amendment_implementation_authorization
amended_implementation_review
exact_v2_rerun_authorization
reviewed_runtime_dependency_closure
run_history_ledger
run_admission
static_input_replay
prior_producer_chain_replay
checkpoint_and_source_identity
disk_revalidation
terminal_exclusivity
staging_topology
producer_extraction
verification_extraction
projection_equality
held_label_invariance
per_fit_environment
retrospective
baseline_evaluator
candidate_evaluator
global_resource_caps
candidate_canonical_coverage
candidate_publication_preconditions
```

No check may be omitted, renamed, duplicated, reordered, or replaced by an
aggregate boolean. `candidate_publication_preconditions` verifies deterministic
candidate-gate bytes and the exact private transaction before rename; it does
not claim the later publication succeeded.

## Shared resource and disk limits

The producer attempts and verification attempt share one global parent budget;
nothing resets for the second extraction:

- cumulative active runtime remains at most `43_200` seconds;
- cumulative resource sample ordinal remains at most `21_600` on the same
  `t=2,4,...,43_200` schedule;
- cumulative canonical resource-log bytes across all producer and verification
  logs remain at most `16_777_216` bytes; and
- a sustained three-sample breach across a process boundary remains a breach.

The verification log uses the distinct exact schema
`agu.vru-causal-verification-resource-sample.v1`. Each row has exactly
`schema_version`, `attempt_role=independent_empty_state_verification`,
`verification_ordinal=1`, `attempt_sample_ordinal`,
`cumulative_sample_ordinal`, `scheduled_cumulative_active_seconds`,
`observed_attempt_active_nanoseconds`, `system_memory_percent`,
`available_memory_bytes`, `free_swap_bytes`, `system_cpu_percent`,
`consecutive_breach_count`, and `breached_limits`. Types, limit names, JSONL
encoding, schedule, and breach semantics are identical to the parent resource
row. The verification attempt-local ordinal starts at one and the global
cumulative ordinal continues from the terminal producer attempt. No producer
`attempt_ordinal` is assigned to verification. The verification attempt records
starting and ending cumulative counters. Both workers must be reaped.

Disk accounting is per physical filesystem. The output-root filesystem reserves
at least `1_350_565_888` new bytes before enforcing the unchanged `3.5 GiB`
post-write reserve: the parent `1_073_741_824` allowance plus four simultaneously
relevant `67_108_864`-byte ceilings for verification embedding, verification
attempt record, root admission, and the eventual result or failure record, plus
the one simultaneously live `8_388_608`-byte transaction-local worker payload. The
consumption-registry filesystem separately reserves `16_777_216` new bytes and
the authorized candidate-bundle filesystem separately reserves `2_097_152` new
bytes, each before its own unchanged `3.5 GiB` post-write reserve. When two or
three targets share one filesystem their simultaneous allowances are summed;
thus the all-on-one-filesystem lower bound is `1_369_440_256` bytes. Producer
and verification logs share the parent's single global `16_777_216` cap and
are not counted twice. Actual retained prior attempts and pre-existing
authorization/review receipts are already reflected in free space and are not
subtracted twice. Admission preflight knows all three authorized paths and
checks this allocation without writing. The external bundle sealer repeats its
filesystem reserve, path, alias, inode and ancestor checks immediately before
opening a stage. Every state-specific calculation may be stricter but never
lower than these simultaneous per-filesystem bounds.

Immediately before and after the verification worker, the supervisor replays
the plan, TASK-0257 graph, checkpoint descriptor, source-video full hashes,
resume CAS, module flock, cumulative counters, disk reserve, and target
absence. A downstream worker/computation/resource failure after attempt
admission publishes `terminal_failure_v2` only while the global trust spine
still verifies; a trust-spine failure is the no-write refusal above. Neither
path publishes a candidate.

## Failure generation before candidate publication

`terminal_failure_v2` is one no-clobber transaction. It always contains
`mechanical_failure.json`. Previously published recoverable producer attempt
directories remain immutable at `attempts/attempt-000N/`; they are never moved
or copied into the failure directory and are replayed only through their
caller-frozen external receipts. When producer extraction reached completion,
the failure directory contains the not-previously-published terminal producer
attempt record and log. If verification was admitted, it additionally contains
the current verification attempt record and complete log, including a
terminal-failure record when verification failed. It contains the producer
embedding only if that artifact verified, and the verification embedding only
if it verified before failure. A verified retrospective may be retained, but no
baseline/candidate evaluator bytes or evaluator receipt appears in any
mechanical-failure generation. Thus
`producer_attempt_chain` cites either an immutable externally receipted prior
attempt directory or the terminal producer record present in this generation.

`agu.vru-causal-temporal-mechanical-failure.v2` has exactly the common false
eligibility fields plus:

```text
provider_slots
producer_attempt_chain
failed_phase
ordered_check_results
resource_summary
publication_summary
decision = mechanical_failure
stop_reason
conditional_downstream
artifact_sha256
```

Provider slots are ordered as parent spec approval, amendment implementation
approval, amended implementation review, exact v2 rerun authorization, run
history ledger, run admission, static inputs, producer attempt,
producer embedding, verification attempt, verification embedding, and
retrospective. Each slot is
exactly `{provider,verification_state,receipt}` with state `verified`, `failed`,
or `not_reached`; only `verified` has a receipt. Every trust-spine provider slot
must be `verified` for this artifact to exist; `failed/not_reached` are
permitted only for downstream worker, embedding, or retrospective providers
reached after the trust spine. Evaluator failures are represented
only by `failed_phase` and the one false ordered check, never by an evaluator
provider slot or receipt.

`failed_phase` is exactly `producer_attempt_1`, `producer_attempt_2`,
`producer_attempt_3`, `pre_verification`,
`verification_before_embedding`, `verification_after_embedding`,
`projection_equality`, `held_label_invariance`, `per_fit_environment`,
`retrospective`, `baseline_evaluator`, `candidate_evaluator`,
`global_resource_caps`, or `candidate_sealing`.
`stop_reason` is exactly `resource_breach`, `runtime_cap`, `sample_cap`,
`log_byte_cap`, `decode_failure`, `model_failure`, `determinism_failure`,
`non_advancing_prefix`, `attempt_limit`,
`external_sigint`, `external_sigterm`, `unsupported_signal`, or
`computation_failure`. The phase/check/reason relation is closed by this table;
the false check is always the one shown for that phase in the following check
prefix table:

| Failed phase | Exact allowed stop reasons |
| --- | --- |
| `producer_attempt_1`, `producer_attempt_2` | `resource_breach`, `runtime_cap`, `sample_cap`, `log_byte_cap`, `decode_failure`, `model_failure`, `determinism_failure`, `non_advancing_prefix`, `external_sigint`, `external_sigterm`, `unsupported_signal` |
| `producer_attempt_3` | all preceding producer reasons plus `attempt_limit` |
| `pre_verification` | `computation_failure`, `external_sigint`, `external_sigterm`, `unsupported_signal` |
| `verification_before_embedding`, `verification_after_embedding` | `resource_breach`, `runtime_cap`, `sample_cap`, `log_byte_cap`, `decode_failure`, `model_failure`, `determinism_failure`, `external_sigint`, `external_sigterm`, `unsupported_signal` |
| `projection_equality`, `held_label_invariance`, `per_fit_environment`, `retrospective`, `baseline_evaluator`, `candidate_evaluator`, `candidate_sealing` | `computation_failure`, `external_sigint`, `external_sigterm`, `unsupported_signal` |
| `global_resource_caps` | `resource_breach`, `runtime_cap`, `sample_cap`, `log_byte_cap`, `external_sigint`, `external_sigterm`, `unsupported_signal` |

For a verification-attempt record, `resource_breach`, `runtime_cap`,
`sample_cap`, `log_byte_cap`, `decode_failure`, `model_failure`,
`determinism_failure`, `external_sigint`, `external_sigterm`, and
`unsupported_signal` map one-to-one to the same outer stop reason and to
`verification_before_embedding` or `verification_after_embedding` according to
the exact embedding-slot state. Its `receipt_failure`, `schema_failure`,
`input_identity_failure`, and `disk_failure` are global trust-spine refusals and
produce no outer failure artifact. Its `publication_failure` leaves untrusted
residue and likewise permits no subsequent publication. No other inner reason,
outer reason, phase, false check, or attempt disposition pairing is accepted.
`non_advancing_prefix` requires a producer attempt with zero new rows;
`attempt_limit` requires attempt 3 to add at least one verified row yet end with
fewer than 45 verified rows, so the two reasons are mutually exclusive.
Every `ordered_check_results` starts with the first thirteen trust checks from the
candidate sequence above, all true, followed by exactly the succeeded prefix
and one false row in this table:

| Failed phase | Additional true checks | One false check |
| --- | --- | --- |
| `producer_attempt_1` | none | `producer_attempt_1` |
| `producer_attempt_2` | none | `producer_attempt_2` |
| `producer_attempt_3` | none | `producer_attempt_3` |
| `pre_verification` | `producer_extraction` | `pre_verification` |
| `verification_before_embedding` | `producer_extraction` | `verification_before_embedding` |
| `verification_after_embedding` | `producer_extraction`, `verification_extraction` | `verification_after_embedding` |
| `projection_equality` | `producer_extraction`, `verification_extraction` | `projection_equality` |
| `held_label_invariance` | candidate-sequence prefix `producer_extraction` through `projection_equality` | `held_label_invariance` |
| `per_fit_environment` | candidate-sequence prefix `producer_extraction` through `held_label_invariance` | `per_fit_environment` |
| `retrospective` | candidate-sequence prefix `producer_extraction` through `per_fit_environment` | `retrospective` |
| `baseline_evaluator` | candidate-sequence prefix `producer_extraction` through `retrospective` | `baseline_evaluator` |
| `candidate_evaluator` | candidate-sequence prefix `producer_extraction` through `baseline_evaluator` | `candidate_evaluator` |
| `global_resource_caps` | candidate-sequence prefix `producer_extraction` through `candidate_evaluator` | `global_resource_caps` |
| `candidate_sealing` | candidate-sequence prefix `producer_extraction` through `candidate_canonical_coverage` | `candidate_publication_preconditions` |

Each named prefix expands literally and may not be encoded as shorthand in the
artifact. No later row is present. The false check, failed phase, and stop reason
must be the unique allowed combination above. `resource_summary` is exactly
`{cumulative_active_runtime_nanoseconds,cumulative_resource_samples,cumulative_resource_log_bytes}`
with nonnegative integers. `publication_summary` is exactly
`{target:terminal_failure_v2,publication_state:not_yet_observed}` and does not
attest to its future rename.

Directory membership is fixed by this table; `mechanical_failure.json` is in
every row:

| Failed phase | Other exact members |
| --- | --- |
| `producer_attempt_1` | `producer_attempt/resource_guard.jsonl`, `producer_attempt/attempt_record.json` |
| `producer_attempt_2` | `producer_attempt/resource_guard.jsonl`, `producer_attempt/attempt_record.json` |
| `producer_attempt_3` | `producer_attempt/resource_guard.jsonl`, `producer_attempt/attempt_record.json` |
| `pre_verification` | `terminal_attempt/resource_guard.jsonl`, `terminal_attempt/attempt_record.json`, `producer_tiled_swin_embeddings.json` |
| `verification_before_embedding` | all `pre_verification` members plus `verification_attempt/resource_guard.jsonl`, `verification_attempt/attempt_record.json` |
| `verification_after_embedding` | all `verification_before_embedding` members plus `verification_tiled_swin_embeddings.json` |
| `projection_equality`, `held_label_invariance`, `per_fit_environment`, or `retrospective` | all `verification_after_embedding` members |
| `baseline_evaluator` | all `retrospective` members plus `temporal_retrospective.json` |
| `candidate_evaluator` | all `retrospective` members plus `temporal_retrospective.json` |
| `global_resource_caps` or `candidate_sealing` | all `retrospective` members plus `temporal_retrospective.json` |

The public failure verifier's `expected_failure_member_receipts` contains every
member in that phase row, including `mechanical_failure.json`, in lexical
relative-path order and uses the exact `GenerationMemberReceipt` union defined
below. Its keys must equal the physical directory listing; no discovery-derived
expected value, basename-only alias, omission, extra, or duplicate is accepted.

Prior recoverable attempts stay under `attempts/` and appear only through their
external six-field receipts. Provider states describe retained provider bytes,
not the failure subject. Every member present in the exact row above must have a
`verified` slot and receipt. A provider whose computation failed before valid
bytes existed is `failed` with null receipt; every later provider is
`not_reached`. A signal or inter-phase check failure may therefore have all
reached providers `verified` and later providers `not_reached`, with no
`failed` provider slot; its one false `ordered_check_results` row is the exact
failure subject. In a producer phase the terminal attempt record/log are
verified while producer embedding is `failed` or `not_reached`. In
`pre_verification` producer attempt and embedding are verified. In
`verification_before_embedding` the verification terminal attempt/log are
verified while verification embedding is failed/not reached. In
`verification_after_embedding` both verification providers are verified. In
the projection/invariance/environment/retrospective phases, all prior providers
are verified and retrospective is failed or not reached as the phase dictates.
In either evaluator, global-cap, or candidate-sealing phase, retrospective is
verified and no evaluator provider is stored. The public
verifier checks this exact phase/member/state matrix and every external attempt
link. Any publication failure leaves only untrusted staging and must not
fabricate a terminal receipt.

## Post-publication verification

Candidate publication and post-publication verification are deliberately two
phases. The producer publishes and fsyncs `candidate_v2`, releases the output
root flock, and returns no trusted result. An external caller then reads the ten
immutable candidate members and seals
`agu.vru-causal-temporal-candidate-receipt-bundle.v1` outside the output root.
That exact artifact has the common false fields plus exactly:

```text
authorization_receipt
run_identity_receipt
run_admission_receipt
static_input_contract
candidate_published_history_head_receipt
candidate_generation_name = candidate_v2
ordered_member_receipts
observed_at_utc
artifact_sha256
```

Authorization and run admission are exact `ArtifactFileReceipt` values; run
identity and candidate-published head are exact `HistoryArtifactReceipt`
values; `static_input_contract` is the exact `StaticInputContractReceipt`.
Together they bind the same
authorization/run/admission/history/input tuple as the candidate. Member rows
have exactly `relative_path`, `receipt_kind`, `size_bytes`,
`internal_sha256_field`, `artifact_sha256`, and `file_sha256`. They occur in the
literal ten-member candidate order above. `relative_path` is that exact safe POSIX-relative path;
`size_bytes` is a nonnegative integer; `file_sha256` is lowercase. For the two
resource JSONL members, `receipt_kind=file_only`, `internal_sha256_field=null`,
and `artifact_sha256=null`.
For the other eight canonical JSON members, `receipt_kind=json` and
`internal_sha256_field` is the provider's exact nonempty safe field name and
`artifact_sha256` is a lowercase SHA-256 that verifies that exact field. This
row shape is also the exact `GenerationMemberReceipt` used for failure-member
verification. No missing, extra, duplicate, reordered, aliased, or coerced
row is accepted. The timestamp is UTC RFC3339. Its own external file SHA-256 is
frozen by the caller. The bundle does not claim computational validity.

The external sealer accepts only the authorization-bound absent bundle path.
It first takes the output-root flock and a no-follow exclusive flock on the
existing bundle parent, replays the complete externally receipted history and
candidate, checks output/bundle/registry disjointness and the bundle filesystem
reserve, then writes compact canonical bytes through the one fixed mode-`0o600`
no-follow stage whose basename is exactly
`.<authorization-artifact-sha256>.<bundle-final-basename>.task0258-bundle-stage`,
followed by file-fsync, no-clobber rename and parent-fsync. The authorization
hash is all 64 lowercase characters and the final basename is the exact safe
basename already bound by the authorization; no random or caller-selected
stage name exists. Before writing, both final and this exact stage must be
absent. An exact-stage residue is a stable no-write refusal and is never
repaired; unrelated siblings are ignored only after no-follow alias/ancestor/
inode checks prove they are not either owned path. After rename, the sealer
re-opens the published bytes before returning no capability. A later existing
final is always a no-write refusal for this sealer; its manifest/API deliberately
has no existing-final receipt variant. The caller must independently freeze the
new result's internal/file receipt pair and can continue only through
`load_verified_candidate_receipt_bundle` or the postverification operation.
Thus a crash before rename leaves the one mechanically identifiable stage,
while a crash after rename leaves only the exact final for external freeze and
read-only loading; neither is resealed, repaired or mistaken for an unrelated
parent-directory child.

A separate verification invocation receives the bundle path plus external
internal/file receipts, takes the same no-follow nonblocking exclusive output
root flock, and replays the bundle and candidate bytes. Candidate drift while
the lock was released fails closed. From this first locked replay until result
or postverification-failure publication, one flock remains held. Under that
lock, the verifier:

1. replays the user-approved plan and complete TASK-0257 inputs;
2. replays every externally supplied recoverable producer attempt directory,
   resource log and resume CAS, plus the terminal producer attempt;
3. verifies exact candidate member coverage and caller-supplied internal/file
   receipts for every member;
4. loads producer and verification embeddings through their distinct
   role-bound capabilities, rejects path/inode aliases, and compares their
   exact computational projections;
5. replays the verification attempt, its provider-bound read-isolation
   policy/attestation and globally cumulative resource limits, including exact
   zero reads of every prior resume and producer-vector-bearing path;
6. independently reruns held-label invariance, the retrospective, both
   baseline variants, the sole candidate variant, every per-fit environment
   check, and both all-45 refits;
7. applies the phase-specific topology check defined below; and
8. derives the frozen metric outcome and terminal decision.

The pre-rename topology is exactly `candidate_verification_pending`: candidate
exists; `terminal_failure_v2`, `verified_result_v2`, and
`postverification_failure_v2` are absent. During result publication, exactly one
active staging directory is allowed: it must be absent before creation, opened
no-follow under the held root flock, and bound by a private transaction
capability recording its resolved parent, basename, `(device,inode)`, expected
single member, and exact staged bytes. Immediately before rename, repeat all
receipts, identities, aliases and topology checks; require that bound active
stage and reject every other `.task0258-*` entry. Publish that stage by
same-filesystem `RENAME_EXCL`, fsync the result directory and parent, then keep
the lock while re-opening the published result.

The post-rename and future-consumer topology is instead exactly
`verified_terminal_result`: candidate and the same `verified_result_v2` exist;
`terminal_failure_v2`, `postverification_failure_v2`, and every staging entry
are absent.
Re-open and verify the registry's exact identity, internal SHA, canonical bytes,
member receipts and computation under that topology. A post-rename verification
failure mints no capability and cannot be converted into a success by the
result's mere presence; it does not publish a contradictory sibling failure.
The untrusted result directory remains fail-closed for operator diagnosis. A
future consumer takes the same lock and repeats the common computational checks
and this post-rename topology before minting or using the result capability.

This contract trusts the reviewed supervisor and the external caller that
freezes receipts after publication. A `(Path, expected hashes)` API cannot
cryptographically prove where the expected strings came from. It guarantees
that bytes cannot be changed or self-resealed after the caller-frozen receipt,
and it rejects wrong timing/state, same path/inode, whole-artifact copy, role
substitution, bypass of the worker path, and later drift. It does not claim to
detect malicious reviewed code that copies producer vectors into a newly
wrapped verification artifact. A future stronger threat model would require an
independent signing authority and a new specification.

## Exact post-publication targets

On successful verification, publish exactly:

```text
verified_result_v2/
  verification_registry.json
```

`agu.vru-causal-temporal-postpublication-verification.v2` has the common false
eligibility fields plus exactly:

```text
candidate_generation_name = candidate_v2
authorization_receipts
run_history_contract_receipt
run_admission_receipt
static_input_contract
candidate_receipt_bundle_receipt
candidate_member_receipts
prior_attempt_receipts
producer_embedding_receipt
verification_embedding_receipt
verification_attempt_receipt
ordered_check_results
error_bound_result
decision
stop_reason
evaluator_receipts
publication_observation
conditional_downstream
artifact_sha256
```

`authorization_receipts` binds parent approval, amendment implementation
approval, amended implementation review, and the exact v2 rerun authorization.
`run_history_contract_receipt` is the exact `RunHistoryContractReceipt` at
candidate-published head, and `run_admission_receipt` is its exact
`ArtifactFileReceipt`.
`static_input_contract` is the exact `StaticInputContractReceipt` shared by the
authorization, admission, candidate, and bundle.
`candidate_receipt_bundle_receipt` binds the external bundle's internal and file
SHA-256.
`candidate_member_receipts` has the exact ten member paths listed above, each
using the same `GenerationMemberReceipt` rows as the external bundle.
`prior_attempt_receipts` is an ordinal-ordered list of exact
`{attempt_ordinal,attempt_record_receipt,resource_log_receipt,resume_receipt}`
rows for the zero, one, or two recoverable producer attempts. Ordinal is a
non-boolean integer starting at one without gaps; attempt and resume are parent
six-field `StoredArtifactReceipt`; resource log is parent `FileReceipt`. All
three filenames and hashes match the same attempt directory and its history
marker. Producer embedding, verification embedding, and verification attempt
fields are the exact `GenerationMemberReceipt` rows for their corresponding
candidate members. `evaluator_receipts` is exactly
`{baseline: BASELINE_ROW,candidate: CANDIDATE_ROW}`, where both values are their
exact `GenerationMemberReceipt` rows and equal the two candidate-member rows.

`ordered_check_results` is this exact safe-enum sequence:

```text
parent_spec_authorization
amendment_implementation_authorization
amended_implementation_review
exact_v2_rerun_authorization
reviewed_runtime_dependency_closure
run_history_ledger
run_admission
static_input_replay
prior_producer_chain_replay
checkpoint_and_source_identity
producer_extraction
verification_extraction
projection_equality
held_label_invariance
per_fit_environment
retrospective
baseline_evaluator
candidate_evaluator
global_resource_caps
disk_revalidation
candidate_canonical_coverage
candidate_receipt_bundle
prerename_terminal_exclusivity
bound_active_stage_only
result_publication_preconditions
frozen_error_bounds
```

All except the final error-bound row must pass for a result artifact to exist.

`decision` is `mechanical_pass` when every row including error bounds passes,
or `temporal-hypothesis-rejected` when only the error-bound row fails.
`stop_reason` is null for pass and `temporal-hypothesis-rejected` otherwise.
`publication_observation` is exactly
`{candidate_visible:true,result_publication_state:not_yet_observed,bound_active_stage_is_only_stage:true}`.
It makes no claim about the future rename, post-rename exclusivity, or residue
absence. Those facts are established only after rename by locked re-open, the
durable `verified_result_published` history marker with root CAS, and the later
caller-frozen result/history receipts. They are required before
`VerifiedModuleAResult` can be minted but are never written back into the
prepublication registry bytes.
For `mechanical_pass`, `conditional_downstream` keeps runtime/formal/promotion
authority false and permits only the statement that a later separately
specified and user-approved Module B may inspect the opaque verified result.
For `temporal-hypothesis-rejected`, every conditional downstream field is false
and the module stops.

The same global trust spine and explicit parent supersession defined above
apply here, extended by candidate canonical coverage and the external candidate
receipt bundle. Any replay failure is a receipt-free no-write refusal: it
returns nonzero, publishes neither result nor failure, and appends no history.
A stale previously minted capability is invalid. This rule takes precedence
over every failure-publication clause.

Only while the entire trust spine remains verified may a non-metric
computational or global-resource check fail by publishing exactly:

```text
postverification_failure_v2/
  failure.json
```

The exact postverification failure allowlist is the common false fields plus:

```text
dependency_provider_slots
failed_check_name
observed_result
decision = mechanical_failure
stop_reason
final_result_published = false
conditional_downstream
artifact_sha256
```

`dependency_provider_slots` is ordered as parent spec approval, amendment
implementation approval, amended implementation review, exact v2 rerun
authorization, run history ledger, run admission, static inputs,
each prior producer attempt record/log/resume in increasing
ordinal, and the external candidate receipt bundle. Providers that do not
exist for the frozen attempt count are omitted by the exact chain shape; every
listed provider occurs exactly once. The verifier independently replays all ten
candidate members from the external bundle, but `failure.json` stores only the
bundle's `ArtifactFileReceipt`, not its member rows. It therefore contains no
baseline/candidate evaluator receipt, directly or as a role-qualified member,
while still binding the immutable caller-frozen candidate as a whole.

Every slot is exactly `{provider,verification_state=verified,receipt}` with its
non-null role-appropriate external receipt. A missing, failed, not-reached,
drifted, aliased, unparsable, or schema-invalid dependency is a trust-spine
no-write refusal and therefore is not representable in a published failure.

`failed_check_name` is exactly one of `producer_extraction`, `verification_extraction`,
`projection_equality`, `held_label_invariance`, `per_fit_environment`,
`retrospective`, `baseline_evaluator`, `candidate_evaluator`,
or `global_resource_caps`.
`stop_reason` equals that value. `observed_result` has exactly
`{subject_kind=check,subject,observation_state}` with subject equal to
`failed_check_name`. `producer_extraction`, `verification_extraction`,
`projection_equality`, `held_label_invariance`, `per_fit_environment`,
`retrospective`, `baseline_evaluator`, and `candidate_evaluator` map only to
`computation_mismatch`; `global_resource_caps` maps only to `limit_failure`.
No other check/subject/state combination is valid. The observation carries no expected
or observed hash and therefore does not bless unverified bytes. It never
contains an evaluator authority receipt.
The success and failure targets are mutually exclusive. The failure artifact
does not claim that its own publication was previously observed; its external
file receipt and locked re-open establish that boundary.

## Public capability API

The following seven functions are review-sandbox-only. They are not exported by
any production CLI, and their return values are diagnostic rather than
production capabilities:

```python
def bind_implementation_review_discovery_context(
    *,
    expected_check_name: Literal["focused_pytest", "full_pytest"],
    expected_command_sha256: str,
) -> VerifiedReviewDiscoveryContext: ...


def bind_implementation_review_sandbox_context(
    *,
    expected_check_name: Literal["focused_pytest", "full_pytest"],
    expected_command_sha256: str,
) -> VerifiedImplementationReviewSandboxContext: ...


def replay_module_a_read_traversal_for_discovery(
    *,
    discovery_context: VerifiedReviewDiscoveryContext,
    operation: ModuleAV2Operation,
    operation_input_manifest_path: Path,
    expected_manifest_artifact_sha256: str,
    expected_manifest_file_sha256: str,
) -> DiscoveryOnlyObservation: ...


def bind_synthetic_module_a_discovery_context(
    *,
    discovery_context: VerifiedReviewDiscoveryContext,
    isolated_temp_ancestor: Path,
) -> VerifiedSyntheticDiscoveryTransactionContext: ...


def bind_synthetic_module_a_transaction_context(
    *,
    review_sandbox: VerifiedImplementationReviewSandboxContext,
    isolated_temp_ancestor: Path,
) -> VerifiedSyntheticModuleATransactionContext: ...


def exercise_module_a_v2_state_machine_for_discovery(
    *,
    synthetic_context: VerifiedSyntheticDiscoveryTransactionContext,
    scenario: SyntheticModuleAScenario,
) -> SyntheticModuleAObservation: ...


def exercise_module_a_v2_state_machine_for_review(
    *,
    synthetic_context: VerifiedSyntheticModuleATransactionContext,
    scenario: SyntheticModuleAScenario,
) -> SyntheticModuleAObservation: ...
```

The discovery binder consumes FDs 193/194/195/196 and, on fixed-point
validation iterations, FD 197; the evidence binder consumes FDs 197/198/199.
For executable checks it additionally consumes the verified executable FD 200
and retained-runtime-root FD 201; both must be closed for `diff_check`. Each
uses its distinct one-bind protocol. The traversal function accepts only
the six closed manifest schemas and invokes the same private read/parse/path
primitives and exact loader DAG as the corresponding production operation, but
cannot mint a capability or enter its target. The two synthetic
binders reopen their own external-runner-bound temporary ancestor and require
its exact physical identity. The discovery dispatcher is the same fixed
scenario engine but its observations are diagnostic/audit-only and can never
be sealed as check evidence. The evidence dispatcher calls only the shared
production primitives named above. None accepts real artifacts, production
paths, a model/device selector, or a production authorization; every discovery
or review type is rejected by every production function below.

```python
def load_verified_parent_module_a_spec_approval(
    *,
    execution_context: VerifiedImplementationReviewSandboxContext | VerifiedAuthorizedRerunBootstrapContext,
    approval_path: Path,
    expected_artifact_sha256: str,
    expected_file_sha256: str,
    approved_spec_paths: Mapping[str, Path],
    expected_fresh_review_internal_sha256: str,
    expected_fresh_review_file_sha256: str,
    expected_approval_statement_sha256: str,
) -> VerifiedParentModuleASpecApproval: ...


def load_verified_producer_tiled_swin_embeddings(
    *,
    embeddings_path: Path,
    expected_artifact_sha256: str,
    expected_file_sha256: str,
    implementation_approval: VerifiedAmendmentImplementationApproval,
    rerun_authorization: VerifiedModuleAV2RerunAuthorization,
    run_history: VerifiedRunHistoryLedger,
    run_admission: VerifiedModuleAV2RunAdmission,
    static_inputs: VerifiedModuleAStaticInputs,
) -> VerifiedProducerTiledSwinEmbeddings: ...


def load_verified_amendment_implementation_approval(
    *,
    execution_context: VerifiedImplementationReviewSandboxContext | VerifiedAuthorizedRerunBootstrapContext,
    repository_root: Path,
    approval_path: Path,
    expected_artifact_sha256: str,
    expected_file_sha256: str,
    parent_approval: VerifiedParentModuleASpecApproval,
    amendment_path: Path,
    expected_amendment_file_sha256: str,
    amendment_review_path: Path,
    expected_amendment_review_artifact_sha256: str,
    expected_amendment_review_file_sha256: str,
    implementation_scope_baseline_path: Path,
    expected_implementation_scope_baseline_artifact_sha256: str,
    expected_implementation_scope_baseline_file_sha256: str,
) -> VerifiedAmendmentImplementationApproval: ...


def load_verified_module_a_v2_rerun_authorization(
    *,
    bootstrap_context: VerifiedAuthorizedRerunBootstrapContext,
    repository_root: Path,
    authorization_path: Path,
    expected_artifact_sha256: str,
    expected_file_sha256: str,
    implementation_approval: VerifiedAmendmentImplementationApproval,
    implementation_review_path: Path,
    expected_implementation_review_artifact_sha256: str,
    expected_implementation_review_file_sha256: str,
    implementation_scope_baseline_path: Path,
    expected_implementation_scope_baseline_artifact_sha256: str,
    expected_implementation_scope_baseline_file_sha256: str,
    approved_code_paths: Mapping[str, Path],
    approved_test_paths: Mapping[str, Path],
    approved_runtime_dependency_paths: Mapping[str, Path],
    approved_check_configuration_paths: Mapping[str, Path],
    approved_check_input_paths: Mapping[str, Path],
    approved_check_output_paths: Mapping[str, Path],
    static_inputs: VerifiedModuleAStaticInputs,
    output_root: Path,
    candidate_receipt_bundle_path: Path,
) -> VerifiedModuleAV2RerunAuthorization: ...


def load_verified_module_a_static_inputs(
    *,
    execution_context: VerifiedImplementationReviewSandboxContext | VerifiedAuthorizedRerunBootstrapContext,
    temporal_plan_path: Path,
    expected_temporal_plan_artifact_sha256: str,
    expected_temporal_plan_file_sha256: str,
    task0257_input_paths: Task0257InputPaths,
    expected_task0257_receipts: Task0257ExpectedReceipts,
    expected_task0257_receipts_projection_sha256: str,
) -> VerifiedModuleAStaticInputs: ...


# Sole production bootstrap target for first admission. It calls the two
# lower-level functions below in one process without exposing the preflight
# token or releasing either lock.
def module_a_v2_preflight_and_publish_admission(
    *,
    implementation_approval: VerifiedAmendmentImplementationApproval,
    rerun_authorization: VerifiedModuleAV2RerunAuthorization,
    static_inputs: VerifiedModuleAStaticInputs,
    checkpoint_path: Path,
    expected_checkpoint_sha256: str,
    source_video_paths: Mapping[str, Path],
    expected_source_video_sha256s: Mapping[str, str],
    output_root: Path,
) -> None: ...


def preflight_module_a_v2_run_admission(
    *,
    implementation_approval: VerifiedAmendmentImplementationApproval,
    rerun_authorization: VerifiedModuleAV2RerunAuthorization,
    static_inputs: VerifiedModuleAStaticInputs,
    checkpoint_path: Path,
    expected_checkpoint_sha256: str,
    source_video_paths: Mapping[str, Path],
    expected_source_video_sha256s: Mapping[str, str],
    output_root: Path,
) -> VerifiedCompleteNoWritePreflight: ...


def publish_module_a_v2_run_admission(
    *,
    preflight: VerifiedCompleteNoWritePreflight,
) -> None: ...


def recover_module_a_v2_run_admission_completion(
    *,
    bootstrap_context: VerifiedAuthorizedRerunBootstrapContext,
    implementation_approval: VerifiedAmendmentImplementationApproval,
    rerun_authorization: VerifiedModuleAV2RerunAuthorization,
    static_inputs: VerifiedModuleAStaticInputs,
    claim_path: Path,
    expected_claim_artifact_sha256: str,
    expected_claim_file_sha256: str,
    admission_path: Path,
    expected_admission_artifact_sha256: str,
    expected_admission_file_sha256: str,
    output_root: Path,
) -> None: ...


def load_verified_run_history_ledger(
    *,
    implementation_approval: VerifiedAmendmentImplementationApproval,
    rerun_authorization: VerifiedModuleAV2RerunAuthorization,
    registry_directory: Path,
    expected_registry_receipts: Sequence[NamedHistoryArtifactReceipt],
) -> VerifiedRunHistoryLedger: ...


def load_verified_module_a_v2_run_admission(
    *,
    implementation_approval: VerifiedAmendmentImplementationApproval,
    rerun_authorization: VerifiedModuleAV2RerunAuthorization,
    run_history: VerifiedRunHistoryLedger,
    static_inputs: VerifiedModuleAStaticInputs,
    admission_path: Path,
    expected_artifact_sha256: str,
    expected_file_sha256: str,
) -> VerifiedModuleAV2RunAdmission: ...


def load_verified_producer_invocation_inputs(
    *,
    implementation_approval: VerifiedAmendmentImplementationApproval,
    rerun_authorization: VerifiedModuleAV2RerunAuthorization,
    run_history: VerifiedRunHistoryLedger,
    run_admission: VerifiedModuleAV2RunAdmission,
    static_inputs: VerifiedModuleAStaticInputs,
    checkpoint_path: Path,
    expected_checkpoint_sha256: str,
    source_video_paths: Mapping[str, Path],
    expected_source_video_sha256s: Mapping[str, str],
    prior_attempt_paths: Sequence[Path],
    expected_prior_attempt_receipts: Sequence[StoredArtifactReceipt],
    resume_path: Path | None,
    expected_resume_cas: ResumeCAS | None,
    device = "mps",
    batch_size = 1,
) -> VerifiedProducerInvocationInputs: ...


def load_verified_verification_tiled_swin_embeddings(
    *,
    embeddings_path: Path,
    expected_artifact_sha256: str,
    expected_file_sha256: str,
    implementation_approval: VerifiedAmendmentImplementationApproval,
    rerun_authorization: VerifiedModuleAV2RerunAuthorization,
    run_history: VerifiedRunHistoryLedger,
    run_admission: VerifiedModuleAV2RunAdmission,
    static_inputs: VerifiedModuleAStaticInputs,
) -> VerifiedVerificationTiledSwinEmbeddings: ...


def load_verified_candidate_receipt_bundle(
    *,
    bundle_path: Path,
    expected_artifact_sha256: str,
    expected_file_sha256: str,
    implementation_approval: VerifiedAmendmentImplementationApproval,
    rerun_authorization: VerifiedModuleAV2RerunAuthorization,
    run_history: VerifiedRunHistoryLedger,
    run_admission: VerifiedModuleAV2RunAdmission,
    static_inputs: VerifiedModuleAStaticInputs,
    output_root: Path,
) -> VerifiedCandidateReceiptBundle: ...


def seal_module_a_candidate_receipt_bundle(
    *,
    bundle_path: Path,
    implementation_approval: VerifiedAmendmentImplementationApproval,
    rerun_authorization: VerifiedModuleAV2RerunAuthorization,
    run_history: VerifiedRunHistoryLedger,
    run_admission: VerifiedModuleAV2RunAdmission,
    static_inputs: VerifiedModuleAStaticInputs,
    output_root: Path,
) -> None: ...


def publish_module_a_candidate(
    *,
    output_root: Path,
    implementation_approval: VerifiedAmendmentImplementationApproval,
    rerun_authorization: VerifiedModuleAV2RerunAuthorization,
    run_history: VerifiedRunHistoryLedger,
    run_admission: VerifiedModuleAV2RunAdmission,
    producer_invocation: VerifiedProducerInvocationInputs,
) -> None: ...


def publish_module_a_postpublication_verification(
    *,
    output_root: Path,
    implementation_approval: VerifiedAmendmentImplementationApproval,
    rerun_authorization: VerifiedModuleAV2RerunAuthorization,
    run_history: VerifiedRunHistoryLedger,
    run_admission: VerifiedModuleAV2RunAdmission,
    candidate_receipt_bundle: VerifiedCandidateReceiptBundle,
    static_inputs: VerifiedModuleAStaticInputs,
    producer_embeddings: VerifiedProducerTiledSwinEmbeddings,
    verification_embeddings: VerifiedVerificationTiledSwinEmbeddings,
    expected_prior_attempt_receipts: Sequence[StoredArtifactReceipt],
) -> None: ...


def verify_module_a_pre_candidate_failure(
    *,
    output_root: Path,
    implementation_approval: VerifiedAmendmentImplementationApproval,
    rerun_authorization: VerifiedModuleAV2RerunAuthorization,
    run_history: VerifiedRunHistoryLedger,
    run_admission: VerifiedModuleAV2RunAdmission,
    static_inputs: VerifiedModuleAStaticInputs,
    failure_path: Path,
    expected_failure_artifact_sha256: str,
    expected_failure_file_sha256: str,
    expected_failure_member_receipts: Sequence[GenerationMemberReceipt],
    expected_prior_attempt_receipts: Sequence[StoredArtifactReceipt],
) -> None: ...


def verify_module_a_postverification_failure(
    *,
    output_root: Path,
    implementation_approval: VerifiedAmendmentImplementationApproval,
    rerun_authorization: VerifiedModuleAV2RerunAuthorization,
    run_history: VerifiedRunHistoryLedger,
    run_admission: VerifiedModuleAV2RunAdmission,
    static_inputs: VerifiedModuleAStaticInputs,
    candidate_receipt_bundle: VerifiedCandidateReceiptBundle,
    failure_path: Path,
    expected_failure_artifact_sha256: str,
    expected_failure_file_sha256: str,
) -> None: ...


def load_verified_existing_module_a_terminal(
    *,
    terminal_kind: Literal[
        "pre_candidate_failure",
        "postverification_failure",
        "verified_result",
    ],
    output_root: Path,
    implementation_approval: VerifiedAmendmentImplementationApproval,
    rerun_authorization: VerifiedModuleAV2RerunAuthorization,
    run_history: VerifiedRunHistoryLedger,
    run_admission: VerifiedModuleAV2RunAdmission,
    static_inputs: VerifiedModuleAStaticInputs,
    candidate_receipt_bundle: VerifiedCandidateReceiptBundle | None,
    terminal_artifact_path: Path,
    expected_terminal_artifact_sha256: str,
    expected_terminal_file_sha256: str,
    expected_terminal_member_receipts: Sequence[GenerationMemberReceipt] | None,
    producer_embeddings: VerifiedProducerTiledSwinEmbeddings | None,
    verification_embeddings: VerifiedVerificationTiledSwinEmbeddings | None,
    expected_prior_attempt_receipts: Sequence[StoredArtifactReceipt],
) -> VerifiedModuleAResult | VerifiedModuleAFailureObservation: ...


def load_verified_module_a_result(
    *,
    output_root: Path,
    implementation_approval: VerifiedAmendmentImplementationApproval,
    rerun_authorization: VerifiedModuleAV2RerunAuthorization,
    run_history: VerifiedRunHistoryLedger,
    run_admission: VerifiedModuleAV2RunAdmission,
    candidate_receipt_bundle: VerifiedCandidateReceiptBundle,
    result_registry_path: Path,
    expected_registry_artifact_sha256: str,
    expected_registry_file_sha256: str,
    static_inputs: VerifiedModuleAStaticInputs,
    producer_embeddings: VerifiedProducerTiledSwinEmbeddings,
    verification_embeddings: VerifiedVerificationTiledSwinEmbeddings,
    expected_prior_attempt_receipts: Sequence[StoredArtifactReceipt],
) -> VerifiedModuleAResult: ...
```

`load_verified_existing_module_a_terminal` first requires exactly one stable
topology and the manifest's exact null/non-null variant. It dispatches
`pre_candidate_failure` only to `verify_module_a_pre_candidate_failure`,
`postverification_failure` only to
`verify_module_a_postverification_failure`, and `verified_result` only to
`load_verified_module_a_result`. The first two return an opaque read-only
`VerifiedModuleAFailureObservation` containing only terminal kind, fixed
`decision=mechanical_failure`, and the externally verified terminal receipt;
it exposes no evaluator receipt and cannot satisfy a result or Module-B
registry lookup. Only the third branch may return `VerifiedModuleAResult`.
Topology/variant mismatch rejects rather than probing another branch.

`VerifiedProducerInvocationInputs` contains the exact parent extractor
parameters, source/checkpoint identities, prior attempt receipts, ResumeCAS,
device and batch contract shown in its loader signature. The candidate
publisher may use no directory discovery, self-derived prior receipt, raw
feature matrix, or hidden process-global state. It runs producer and verification
workers internally and passes only the transaction-local fresh capabilities
defined above into candidate sealing.

All production capability return types listed above are opaque, registered,
exact-type capabilities. The seven review-sandbox functions return separately
registered test-only types that can never satisfy a production registry lookup.
Every public amended producer API and CLI requires the verified
implementation-approval and rerun-authorization capabilities and performs their
zero-write preflight before any other input read or side effect. At every use,
capabilities re-open their bound paths and revalidate file identity, external
receipts, roles, output-root topology, candidate/result state and complete
computation. `VerifiedModuleAResult` exposes only read-only projections of the
terminal decision and evaluator receipts; it is not a public dataclass or raw
mapping. Module B, if separately approved later, must accept and fully
revalidate this complete capability and require its decision to be exactly
`mechanical_pass`; it rejects a hypothesis-rejected result before exposing any
evaluator receipt. A raw decision, registry mapping, or pair of evaluator
receipts is insufficient.

## TDD, review and execution gates

Implementation may start only after this file has a fresh-context review with
Critical/Required `0/0` and the user approves its exact SHA-256. Approval of the
amendment authorizes implementation only. After implementation, a different
fresh reviewer must report Critical/Required `0/0`. Only then may the user
separately authorize one exact `vru_causal_temporal_retrospective_v2` rerun.

Required RED tests cover:

- parent/amendment/review/rerun authorization drift, code/test drift, missing
  full-suite evidence, and reuse at a second root reject before writes;
- the externally receipted pre-edit implementation baseline rejects every
  unapproved new/deleted/renamed/modified member anywhere in the complete
  repository outside exact `.git`/`.venv` exclusions, including a newly read
  `utils/*.json` or other non-executable data file, even if current import/test
  discovery would otherwise add it as green evidence; the
  sole bootstrap creation has the one exact `scripts/` child-set delta, while
  any other directory delta or `st_nlink != 1` rejects;
- run admission requires the complete no-write preflight, publishes only the
  one-file root, returns no public fresh token, and externally receipted resume
  cannot create a second admission;
- moving or deleting the completed output root cannot erase the independent
  claim/completion ledger or double-spend the exact rerun authorization; crash
  recovery may finish only the same claimed nonce;
- candidate gate cannot contain a terminal decision or claim its publication;
- each worker request is sealed before its admitted marker, the marker binds
  only the request hash, and the later nonpersistent claim envelope binds the
  marker receipt without any request/marker receipt cycle; the admitted-stage
  CAS uses the one global seven-field directory row shape;
- verification embedding/attempt sealing follows the acyclic receipt order;
- old v1 or different-run attempts/embeddings reject at every v2 worker,
  candidate and postverification boundary;
- producer and verification artifacts reject same path, inode, role, bytes,
  worker-bypass provenance and receipt drift;
- the external review sandbox and authorized-rerun bootstrap reject cross-mode
  reuse; synthetic transactions escape neither their bound temp ancestor nor
  no-model boundary, share the exact production atomic/state and worker-process
  primitives, and cannot mint or enter any production capability/API; direct production CLI
  invocation without an authorized bootstrap context rejects before reads;
- the exact pytest process mints its review context only once from inherited
  FDs 197/198/199/200/201/206 with matching PID/nonce/command/snapshot/runtime/
  executable identity, while the driver independently consumes FD 204 and
  emits its one receipt on FD 205; swapping/reusing FD-202 request, actual
  FD-203 source, FD-204 attestation, FD-205 receipt, check or phase, or providing
  any script bytes on stdin rejects before
  target import; token
  transfer across exec/fork/spawn, wrong/reused/writable FDs, absent runner
  attestation, or an unreceipted outer
  launcher cannot transfer or mint that context;
- the separate discovery process binds only from FDs 193/194/195/196 plus
  conditional fixed-point FD197, can execute
  diagnostic fixture/synthetic branches without an evidence manifest, and can
  never seal evidence or enter a production API; discovery and fixed-point
  evidence PIDs/nonces/contexts are mutually rejected; the one exact discovery
  traversal API covers all six manifests/loader-DAG endpoints using shared
  read/parse/path primitives and returns only the closed observation schema;
- check evidence rejects wrong repository/snapshot identity, executable FD/CAS or argv,
  any unlisted environment variable including pytest/plugin injection, and any
  check-specific immutable audited snapshot drift; focused and full pytest bind
  every test/conftest/plugin/local-helper/data/subprocess file actually read;
  separate stat/lstat queries, full stat fields including device/inode/time,
  access flags/results, readlink/listdir/xattr arguments/results, negative
  lookups, modes, symlink targets, directory membership and hardlink relations are reproduced
  exactly; mutation of any such namespace fact or a fixed point after three
  iterations fails; a second-pass resolution outside the discovery manifest fails; replacing/restoring
  original repository paths or an executable symlink between receipt and use
  cannot change evidence bytes; direct Python execution contains no zsh-added
  environment; the fixed non-Git whitespace check covers current
  bytes of all eighteen literal paths, so missing, symlinked, empty, untracked
  or whitespace-bad files cannot be hidden by config or another status;
- the four executable review checks launch only through the immutable
  `-P -S` FD-202/203 driver, build package paths from the sealed manifest,
  never execute `.pth`, and dispatch only the exact pytest/Ruff module and
  target argv; both pytest requests include `-p no:cacheprovider
  --import-mode=importlib`, preserve the exact three-row `sys.path` through
  collection/fixtures/exit, and any
  repository write-open, site initialization or path-backed venv entry rejects;
- review and production `sys.path` is exactly stdlib, lib-dynload and sealed
  site-packages in the three frozen rows/order; a missing/reordered row, empty
  or mutable-repository entry, or same-name shadow module outside them rejects;
- source executable device/inode is never compared to a copied image inode;
  source-to-copy size/hash binding, the exact runtime member and FD-200 physical
  CAS are independently mutated in REDs, and each wrong edge rejects;
- only the exact bootstrap loader sites may use the frozen
  `importlib`/`compile`/`exec` exception; the same AST node elsewhere rejects;
  replacing or mutating the bootstrap path between verification and child
  start cannot change the retained bootstrap-source member actually executed
  through `/dev/fd`; wrong source FD/CAS/hash/offset, duplicate channel or
  non-`/dev/null` stdin rejects; replacing the original
  `.venv` symlink/base interpreter or any runtime member between receipt and
  spawn cannot change the retained read-only runtime image; manifest member,
  mount-flag or image-root drift rejects; every alternate path-backed/argv/
  descriptor invocation rejects;
- the runtime manifest's exact base-runtime/venv/site-package/dylib/tool member
  array and projection are independently replayable; missing/extra/reordered,
  wrong-mode, symlink-target, hardlink or executable-root rows reject; the
  complete static Torch/Torchvision/MPS/native tree is permitted even when a
  no-model review check did not read a member, while any member outside that
  independently sealed tree rejects;
- `site.main()`/`.pth` execution is forbidden; every editable/repository
  path/finder/hook and protected preloaded `sys.modules` entry is removed,
  import caches and builtins/importlib hooks equal the closed projection,
  protected module names absent from the immutable snapshot fail terminally,
  and any mutable-repository fallthrough or wrong module loader/spec/source hash
  rejects before target import;
- every closed bootstrap operation dispatches only to its one fixed callable;
  the plan sealer, analysis module, unknown operation, mismatched request
  manifest, repeated illegal phase and caller-selected module/callable reject
  before target import;
- each of the six operation-specific manifests rejects missing/extra/wrong-type
  reconstruction inputs, wrong variant nullability, hidden process-global
  paths, directory discovery and self-derived receipts; the terminal wrapper
  reaches each of the three stable variants, but only the result branch can
  return a result capability;
- the outer bootstrap authorization path/internal/file receipt must equal the
  nested manifest authorization and every duplicated operation/root/bundle/
  registry value before any loader or target import;
- every opaque capability rejects direct construction, `object.__new__`,
  subclassing, copy/deepcopy, pickle/deserialization, token cloning, mutation,
  and use in another run; caller mappings/arrays mutated after wrapper creation
  cannot alter its immutable snapshot, while bound-file drift at use rejects;
- `fork` poisons all inherited registries/tokens and closes Module-A
  descriptors, while spawn/pickle transfer rejects; producer and verification
  children each establish a new PID/epoch through the role-specific bootstrap,
  receive no parent token/lock/output/model/vector (only the fixed
  bootstrap/request/liveness/result/marker/audit-gate/claim-envelope descriptors), and the parent mints a fresh capability
  only after independently replaying their staged bytes;
- each durable admitted marker contains the unique worker-launch claim and
  historical stage CAS; every duplicated PID/source/FD/path/identity field is
  compared with the request and actual process before model initialization;
  wrong/reused secret/request/role/callable/envelope, parent death or liveness
  EOF, a total frame of 65,536 bytes, a 65,537th byte,
  schema/status mismatch, stage replacement and any
  respawn after an unresolved admitted head reject without further heavy work;
  normal child completion closes the result FD and exits explicitly without a
  watchdog/thread-join deadlock, while every role/disposition has the one exact
  staged-receipt/counter/observation/wait-status shape; all PID/FD/liveness and
  status paths run in real temp-bound no-model subprocess REDs;
- the worker request contains no future marker receipt: after request/marker
  publication the parent passes the one no-follow marker open-file description
  on FD7, the FD9 envelope binds its external receipt, and the child/audit compare
  the same identity/bytes without a path open or hash cycle; path-open, wrong FD,
  wrong envelope receipt or marker replacement rejects before target work;
- the externally authenticated provider-control channel is armed before spawn,
  completes prepare/prepared before admitted-marker publication, gates the child
  on FD8, binds policy/request/parent/child PID/instance/nonce,
  observes through child exit, and returns its one capped post-exit frame only to
  the reaping supervisor; prepare failure publishes no marker and removes the
  live owned stage, while each of up to four strictly sequential sessions has a
  fresh instance and an externally replayable frame-hash chain; forged/
  reordered/truncated frames, wrong SCM_RIGHTS, wrong marker/envelope prepared
  hash, permit-before-audit, child access to the control socket or provider EOF leaves
  the admitted head unresolved and seals no attempt;
  the real Darwin RED creates `AF_UNIX/SOCK_STREAM`, sends fragmented and
  coalesced uint64-length frames, and rejects zero/cap+1/short/missing-LF/
  trailing/extra-FD cases; no `SOCK_SEQPACKET` path exists; provider prepare
  independently parses the complete canonical request/policy and contract/
  manifest preimages rather than accepting hashes or supervisor row labels;
- child frames name only the resource log and optional raw worker payload;
  parent post-`waitpid` finalization embeds the consumed request, payload plus provider policy/
  attestation into the attempt, then follows the exact producer attempt→embedding/
  resume or verification embedding→attempt order, deletes the transient payload
  and rejects any child-precomputed final receipt or payload/final mismatch;
  payload bytes at `8_388_608` pass, cap+1 rejects before final-attempt write,
  and the output-root/all-shared-filesystem reserve constants include exactly
  one simultaneously live payload; a maximal request/policy/payload/
  observation/attestation skeleton proves the `44_105_728 <= 67_108_864`
  bound before marker/spawn, while cap+1 or oversized actual request refuses
  without heavy work;
- production parent/producer/verification startup uses `-P -S`, consumes
  `PYTHONHASHSEED=0` with `ignore_environment=0`, and two independent workers
  must report the same exact hash-state projection; `-I`, `-E`, randomized or
  mismatched literal hashes reject before model/device initialization;
- verification predecessor-history REDs prove the child can verify the marker
  chain/receipt projection while the provider-bound OS audit denies every prior
  `attempt-*/resume.json`, producer private-stage, producer embedding,
  prospective candidate producer-member and unlisted output-root read;
  claim/completion/history path rows are complete, exact, ordered and
  contiguous, the completed-private receipt equals the final row, and the
  four-field canonical subject-projection preimage rejects omission, reorder,
  nullability drift or another projection formula;
  the request exposes only exact prior record/log file paths and no attempt
  directory authority; independently mutating producer private bytes is caught
  by the parent's full pre/post CAS replay and never by exposing vectors to the
  child; forged/missing/overflowed/wrong-PID isolation attestations and omitted
  stat/lstat/fstat/access/readlink/getdents/xattr/read/open/mmap/exec events reject at
  attempt, candidate-bundle and result replay; the exact allowed set includes
  run admission, registry claim/completion/prior markers, record/log files and
  the envelope-bound future marker FD, while every other output-root observation
  remains denied; ordinary checkpoint/media/JSON/runtime open→fstat chains and
  synthetic runtime/OS reads plus inherited-directory FD11 succeed, whereas a
  mismatched FD identity or forbidden model/media/output read rejects;
- discovery/fixed-point, every evidence check and runtime-image build use their
  three role-specific process/audit/output/build limits and observations;
  CPU/RSS/process/FD/audit/check-output/builder-log/runtime-image/reserve
  overflow kills and reaps without sealing a check/runtime/implementation-
  review receipt, while the external fresh review is separately governance-
  receipted and size-bounded without inventing local process metrics; the false
  heavy-execution flag is recomputed only from the four externally audited zero
  production-worker/media-or-checkpoint/device-or-service/embedding-publication
  counters under the amendment's explicit definition, not from a claim that no
  CPU-only test constructor ran;
- raw and attestation bytes at their individual maxima plus the one LF produce
  exactly `16_777_216` bytes and pass; raw-cap+1, attestation-cap+1 or final-
  cap+1 rejects; oversized runtime
  manifest/contract and mismatched raw/attestation/final byte observations also
  reject before any immutable check receipt;
- each check first binds its historical temporary raw capture and the approved
  durable-output parent, then proves suffix byte equality to the reopened fixed
  final; swapping either location/inode, deleting raw before final verification,
  writing a repository-relative output, wrong final basename, a second stage or
  a durable-output observation that does not equal the reopened file rejects;
- direct, aliased, import-side-effect and descendant attempts to run a
  production worker, read approved media/checkpoint, access a device/model
  service or publish embeddings are observed by the external sandbox and
  reject, while a bounded CPU-only `weights=None` constructor does not falsify
  the explicitly narrowed heavy-execution statement;
- the standalone heavy-execution policy rejects missing/extra/reordered role,
  protected-path, endpoint, write or event rows, wrong provider build/signature,
  path alias, either wrong exact projection-protocol literal, cross-role write,
  unlisted open flag/operation and policy-file drift; process/event projections are independently
  recomputed from exact rows, and unknown descendants/events or audit overflow
  cannot be represented as an empty successful observation;
- with that policy active, only review-driver FD203 and synthetic-worker FD4
  may resolve through `/dev/fd`, each solely as its attested regular script with
  matching role/parent/offset/source CAS; swapped numbers, other `/dev/fd/*`,
  ordinary device access or a fallback path reject while the canonical launcher
  succeeds;
- discovery audit/temp caps, evidence raw/attestation/final caps, runtime-builder
  log/image/temp/reserve caps and external fresh-review governance/size each use
  only their role-specific observation; injecting a check-only field into build/
  discovery or claiming local process metrics for fresh governance rejects;
  runtime image and contract parents on different devices reject before a stage
  open, while the pre-manifest regular-member byte sum excludes manifest,
  contract, log and temp at decimal-boundary cases and the common-device pre/post
  reserve is independently recomputed;
- admitted/completed-private stage CAS is replayed only on its immediate edge;
  candidate/failure publication binds the same member receipts, consumes the
  stages and proves residue absence, while future result replay never requires
  a deleted private-stage inode; a supervisor transport/protocol failure leaves
  an unresolved head and publishes no fabricated failure artifact;
- producer/pre-verification failures consume exactly one producer stage,
  verification-or-later failures/candidate consume exactly producer plus
  verification stages, and recoverable publication consumes/projects only the
  current producer stage; every wrong phase/count/projection rejects;
- one-float projection drift rejects;
- global runtime/sample/log/disk accounting includes both workers;
- all verification-worker failures are non-resumable and publish no candidate;
- registry publication holds the root flock and repeats all checks immediately
  before rename and after publication;
- candidate publication, external receipt-bundle freeze, and later locked
  verification are distinct phases; synthetic bundle publish plus
  write/fsync/rename crash residue uses the exact production primitive, and
  wrong/missing bundle receipts reject; after a final bundle exists the sealer
  always refuses without writes, while only the external-receipt loader/
  postverify path may accept the crash-after-rename final;
- missing/extra/reordered/self-resealed prior attempts or candidate members
  reject;
- candidate plus terminal failure, result plus postverification failure, or any
  staging residue rejects;
- the sole active bound result stage is allowed only immediately pre-rename and
  every other or post-rename stage rejects;
- registry creation before candidate publication, self-sealed registry, and
  wrong external registry file receipt reject;
- held-label perturbation preserves every non-truth projection;
- every nested role uses its exact receipt type, every check list rejects an
  omitted/reordered/renamed row, and current-path mutation during a concurrent
  at-use replay rejects without changing the snapshot being executed;
- every pre-candidate phase/false-check/stop-reason/attempt-disposition and
  verification-inner-to-outer reason not present in the closed tables rejects;
- synthetic request/claim/envelope bind the already available launch-attestation
  hash rather than a future execution-attestation hash; the exact
  `module:function` target and discovery/evidence execution kind reject every
  same-short-name or cross-kind dispatch; check-input bootstrap map roles index
  the matched namespace receipt path rather than joining the role slug; and
- only a `mechanical_pass` `VerifiedModuleAResult` can cross a future Module-B
  boundary; a hypothesis-rejected result is terminal and exposes no downstream
  evaluator authority; and
- implementation approval alone cannot start a rerun or authorize Module B.

No heavy extraction or versioned rerun is authorized by drafting, reviewing,
or approving implementation of this amendment.
