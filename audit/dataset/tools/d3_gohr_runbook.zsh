#!/bin/zsh
# Gohr D3/D1 execution runbook. ONE STAGE PER INVOCATION:
#
#     zsh audit/dataset/tools/d3_gohr_runbook.zsh <stage>
#
# Stages, in order:  content  tests  check  commit*  anchor  calibrate*  calverify
#                    d0*  d1*  preflight  d3*  verify
# (* = requires typing an exact authorisation phrase at an interactive terminal; default deny.)
#
# Fail-closed rules:
#   * setup errors (repository, interpreter, log directory) exit immediately;
#   * every command's own exit status is recorded and any failure exits;
#   * each stage requires its prerequisite stage's success marker, and every marker
#     records the hashes it was earned for, so a stale marker cannot unlock a later stage;
#   * output directories must not exist; logs are never overwritten;
#   * staging uses the 15 explicit paths below and is checked against them exactly.
# Environment: GOHR_PY (interpreter, default below), GOHR_RUNLOG (log directory).

emulate -L zsh
setopt NO_UNSET PIPE_FAIL

die() { print -u2 "STOP: $*"; exit 1 }

SCRIPT=${0:A}
REPO=${SCRIPT:h:h:h:h}
cd "$REPO" 2>/dev/null || die "cannot enter repository root $REPO"
[[ "$(git rev-parse --show-toplevel 2>/dev/null)" == "$REPO" ]] || die "$REPO is not the repository root"
PY=${GOHR_PY:-$HOME/.venvs/autond-tf215-audit-20261015T050853Z/bin/python}
[[ -x $PY ]] || die "interpreter $PY not found"
export PYTHONDONTWRITEBYTECODE=1
RUNLOG=${GOHR_RUNLOG:-$HOME/gohr_d3_runlogs}
mkdir -p "$RUNLOG" 2>/dev/null || die "cannot create log directory $RUNLOG"
[[ -d $RUNLOG && -w $RUNLOG ]] || die "log directory $RUNLOG is not writable"
STAMP=$(date -u +%Y%m%dT%H%M%SZ) || die "date failed"
STAGE=${1:-}
[[ -n $STAGE ]] || die "usage: zsh $SCRIPT <stage>"

PREREG=audit/targets/gohr/preregistrations/d3_gohr_speck3264_r5.json
FILES=(
  audit/targets/gohr/preregistrations/d3_gohr_speck3264_r5.json
  audit/targets/gohr/preregistrations/d3_gohr_speck3264_r5.RATIONALE.md
  audit/dataset/experiments/d3_gohr_confirmatory.py
  audit/dataset/experiments/d1_gohr_canonical_census.py
  audit/dataset/tools/d0_gohr_attested_build.py
  audit/dataset/tools/d3_gohr_method_validity_calibration.py
  audit/dataset/tools/d3_gohr_postrun_verify.py
  audit/dataset/tools/d3_gohr_runbook.zsh
  audit/dataset/tests/test_d0_gohr_attested_build.py
  audit/dataset/tests/test_d3_gohr_calibration_tool.py
  audit/dataset/tests/test_d3_gohr_preregistration.py
  audit/dataset/tests/test_d3_gohr_confirmatory.py
  audit/dataset/tests/test_d1_gohr_canonical_census.py
  audit/dataset/tests/test_d3_gohr_postrun_verify.py
  audit/dataset/tests/test_d3_gohr_runbook.py
)
EXPECTED_NEW=$(cat <<'EOF'
f77e3af38ceb2f2c2636354c56e957d99b7ae45a90a40e2623413ecd4fa256c1  audit/targets/gohr/preregistrations/d3_gohr_speck3264_r5.RATIONALE.md
5f5515f9d93d165010b32ee15e12f8f79e46d8c9f08fbd4cf909fa7c951f70ca  audit/dataset/experiments/d3_gohr_confirmatory.py
c107b1876c8e427844e779cfa4a7fe0ae7fb923a455059425d3f2b3a3ea3e92f  audit/dataset/experiments/d1_gohr_canonical_census.py
f08bd5ea3d9778e0ee9bc39f01f9ac61f1dd5a47950f182d22af17417df92c3a  audit/dataset/tools/d0_gohr_attested_build.py
5489325fa312994fd6f67e10dc06ec9223d192fcd04b3d12ffb23274e7d38e8d  audit/dataset/tools/d3_gohr_method_validity_calibration.py
02ea89c65b7b39af88c3773d3e11e920a93aa4306caebf49dec4196986d66732  audit/dataset/tools/d3_gohr_postrun_verify.py
264e66c3485682fd0a197c51faaed2789bed5a0a99659df205823df8ac84af80  audit/dataset/tests/test_d0_gohr_attested_build.py
02098d2bd98698210051da9f702d330218623d0056c171bfefd6c7be078c42d8  audit/dataset/tests/test_d3_gohr_calibration_tool.py
849d5b9a788722c9c04330b47b6c406f2aab67aa9b22cc7370b54c6da61e8533  audit/dataset/tests/test_d3_gohr_preregistration.py
2052c257ecb2a1c9f6f641e17d03f5bb0adbf5d3f8929d975c45837a81ccb3b5  audit/dataset/tests/test_d3_gohr_confirmatory.py
ec809e6414c73f994d373d75958fa4b36afc8d42f6b5929b1bf999d92e01ead2  audit/dataset/tests/test_d1_gohr_canonical_census.py
f4e3caad51c506708928984c4e385953fccd257ae308dbb21fa056da7892f666  audit/dataset/tests/test_d3_gohr_postrun_verify.py
5b63e879a185ae46b35346c71ccb305bfeb1151b582685cf9d9d677f49196824  audit/dataset/tests/test_d3_gohr_runbook.py
EOF
)
EXPECTED_PROTECTED=$(cat <<'EOF'
6d1ebe9e1857318642f1a022d18289078f5eaeefb3a5d9268e9d9b36275bfff7  audit/dataset/d3_distribution_statistics.py
30b94415b11707633228cc3b17c5769e5b2ce0b1b30c6ebbf5fdab2e04b8bb0d  audit/dataset/experiments/d3_autond_distribution.py
095ae3255e844ae4ac4a5025b86df32af706ac98a5e0ad19b0d3c70931cf1e4c  audit/dataset/experiments/d3_gohr_distribution_statistics.py
3f9b46ca278658646a3ace4f0423647db0aba3683a661ee03636abe4e7ae8e04  audit/dataset/tools/d3_method_validity_calibration.py
808dedc1aeb0b32ab8dc30fc72710e5c867bb01f2c241be36a931df300a29d32  audit/dataset/tests/test_d3_method_validity.py
3dc242eb9a96242d6fdcfdc8c5632c047701d41340d8bd51869cfb54495c829a  audit/dataset/tests/test_d3_autond_preregistration.py
a70fb48ff5b4ceff13bb61ae09c14e83554a5c073240f055df2145400f0c9a7e  audit/dataset/tests/test_d3_autond_distribution.py
7fc650f5b451850cd461761b0ac117d7052707dd9fec1c045cab4d754b73ea1d  audit/dataset/canonical.py
3c0b7ac5327c68233ac792062f3559232cfc965b8a403312e552bcb2b516b488  audit/dataset/adapters/gohr.py
be81a5dc0f1f8f68a68fdc1bd7884ada90578c634b9d97150251e70732bd9dd1  audit/dataset/adapters/autond.py
59296f7b7990ec3886fc524ff49256005d360eb3748d3b1570351d669acbb8ce  speck.py
b0585971da829f8cd89e2d1e3f85eed600048fcd04bc7d0d32f33ee1cd03c669  train_nets.py
cbd56c4aed8a9304b8ef5600948aa9c1cf456aefa7e623b10d4c1c1be5e3d114  audit/targets/autond/preregistrations/d3_autond_speck3264_r8.json
09d8a105d4ca8f64cfcdb39a97ab8f8be0273901e1c947e4c8009afafef7bd2c  audit/implementation/preregistrations/ii4_impl_gohr_v1.py
ab02d471bff08758b9b2d6e122acc4252e269adada9ffd30cdab5837d759cc8a  audit/implementation/preregistrations/ii-4-gohr-impl-v1.frozen.json
757e62bfebc237f806134ba0e036ee23797048fe891b58a17cb52ce435633ed2  audit/dataset/evidence/d3/d3_gohr_10000000_1000000_1000000_5r.json
8b59754008e6a5e8e760f6e3361316d0c60d128066a2dd6d5aaa5a7dca485e8d  audit/dataset/evidence/d3/d3_gohr_distribution_consistency_10000000_1000000_1000000_5r.json
cffc1d6416b5bd62fb0a4d19358b87ea297afb1cfdecb8925b5c35bc72c0bdca  audit/dataset/evidence/d1/d1_gohr_10000000_1000000_1000000_5r.json
c8b434c26dd6fee91d4b02447df85a093c7af21371d965778daaacd2cbc09603  audit/dataset/evidence/d2_gohr_speck_r6_rerun.json
64deedb75fb1a070c0c79e261992bd42b260ae096332a7d1e1ccf92a73f15e1b  audit/dataset/evidence/d3_method_validity_20261022_c39726e5/d3_method_validity.json
8f1188cf691935edcf87bff1aae01880b373a05f05e9d7b158cfcae26231a126  audit/dataset/evidence/d3_method_validity_20261022_c39726e5/d3_method_validity_run_manifest.json
cd0317f578e78ea1b67ba0f692a51065a81e3b5035f09cc572b48c10bb6fda31  audit/dataset/evidence/d3_autond_speck3264_r8_confirmatory_6974fba6/d3_autond_certificate.json
3b802eeafd73f7da90650447abed658a3fc8d99dca3be76253f8733d37e375fd  audit/dataset/evidence/d3_autond_speck3264_r8_confirmatory_6974fba6/d3_autond_sealed_audit.json
eebec0696020aa69dc69044031118bfd3137ad0aef150623e04e05e19f071a3c  audit/dataset/evidence/d3_autond_speck3264_r8_confirmatory_6974fba6/d3_autond_run_manifest.json
916323cb3b954f6da99326dce4168bacecd41dfdca737fc29b31a33cd0935cf4  audit/dataset/evidence/d0_autond_speck3264_r8_production/d0_canonical_manifest.json
136ee37266135289254d482a3a1c896c2d33a724d34895eadb689930bb3220c9  audit/dataset/experiments/d1_gohr_exact_census.py
c62603281a746e85f0a08def299ea4cfc12cec8725e4e897e343447ebb211c14  audit/dataset/d1_duplicate_detection.py
1237a213263149364f7a1444c6a71d9a8ed6759f4a7b4a41cef384b471cde6e9  audit/dataset/experiments/d4_gohr_label_shuffle.py
5af1aaa4643a61e3e3f8d48dd158c2bfb951078f95552c7daa3846801d7a9070  audit/dataset/d4_controlled_perturbation.py
76ef648fce078338273cd14aeadb6f39b216e497d0cf3b647598a9498e38a322  audit/dataset/evidence/d4/d4_gohr_label_shuffle.json
EOF
)
TESTS=(test_d0_gohr_attested_build test_d1_gohr_canonical_census test_d3_gohr_calibration_tool
       test_d3_gohr_preregistration test_d3_gohr_confirmatory test_d3_gohr_postrun_verify test_d3_gohr_runbook
       test_d1_duplicate_detection test_d1_adversarial_validation test_autond_d1_failclosed
       test_d3_method_validity test_d3_autond_distribution test_d3_autond_preregistration test_d3_invariants
       test_d3_adversarial_validation test_d3_legacy_golden test_autond_adapter test_autond_source_binding
       test_autond_config_identity)

# ------------------------------------------------------------------ helpers
run() {   # run NAME CMD...: own log, the command's own exit status, exit on failure
  local name=$1; shift
  local log=$RUNLOG/${name}_$STAMP.log
  [[ ! -e $log ]] || die "log $log already exists"
  print "== $name"
  "$@" >"$log" 2>&1
  local rc=$?
  tail -n 12 "$log"
  print "== $name EXIT=$rc  log=$log"
  (( rc == 0 )) || die "$name failed (exit $rc)"
}
sha() { shasum -a 256 "$1" | cut -d' ' -f1 }
files_state() { shasum -a 256 "${FILES[@]}" | shasum -a 256 | cut -d' ' -f1 }
mark() { print -r -- "$2" > "$RUNLOG/$1.ok" || die "cannot write marker $1"; print "MARKED $1: $2" }
require() {   # require STAGE EXPECTED_CONTENT
  [[ -f $RUNLOG/$1.ok && "$(<$RUNLOG/$1.ok)" == "$2" ]] || die "prerequisite stage '$1' has not passed for the current state"
}
authorize() { # default deny: an interactive terminal and the exact phrase are required
  local phrase=$1 ans=""
  [[ -t 0 && -t 1 ]] || die "authorisation requires an interactive terminal; nothing was done"
  exec 3</dev/tty 2>/dev/null || die "no controlling terminal; nothing was done"
  print -n "Type exactly '$phrase' to authorise (anything else cancels): "
  read -r ans <&3
  exec 3<&-
  [[ $ans == "$phrase" ]] || die "not authorised; nothing was done"
  print "AUTHORISED: $phrase"
}
post_commit_ids() {
  PC=$(git log -1 --format=%h --abbrev=12 -- $PREREG)
  [[ -n $PC ]] || die "the preregistration is not committed"
  ANCHOR=$(sha $PREREG)
  CAL=audit/dataset/evidence/d3_gohr_method_validity_$PC
  D0=audit/dataset/evidence/d0_gohr_speck_r5_production_$PC
  D1OUT=audit/dataset/evidence/d1_gohr_speck_r5_canonical_$PC
  D3OUT=audit/dataset/evidence/d3_gohr_speck3264_r5_confirmatory_$PC
}
fresh() { [[ ! -e $1 ]] || die "$1 already exists; evidence is never overwritten" }

# ------------------------------------------------------------------ stages
case $STAGE in
content)
  run content $PY -B -c "import json,audit.dataset.d3_distribution_statistics as d3; c=d3.validate_preregistration_content(json.load(open('$PREREG'))); print('CONTENT OK; approved_utc', c['approved_utc'])"
  print "PREREGISTRATION SHA-256 (pre-commit): $(sha $PREREG)"
  mark content "$(sha $PREREG)"
  ;;
tests)
  for f in $TESTS; do
    run test_$f $PY -B -m pytest -p no:cacheprovider -q -rs audit/dataset/tests/$f.py
    grep -Eq '[0-9]+ passed' $RUNLOG/test_${f}_$STAMP.log || die "no passing tests recorded for $f"
    ! grep -q '^SKIPPED' $RUNLOG/test_${f}_$STAMP.log || die "skipped tests in $f"
  done
  mark tests "$(files_state)"
  ;;
check)
  print -r -- "$EXPECTED_NEW" > $RUNLOG/expected_new_$STAMP.sha
  print -r -- "$EXPECTED_PROTECTED" > $RUNLOG/expected_protected_$STAMP.sha
  run check_new_files shasum -a 256 -c $RUNLOG/expected_new_$STAMP.sha
  run check_protected shasum -a 256 -c $RUNLOG/expected_protected_$STAMP.sha
  [[ -z "$(git diff --cached --name-only)" ]] || die "the index is not empty"
  for f in $FILES; do [[ "$(git status --porcelain -- $f)" == "?? $f" ]] || die "$f is not an untracked new file"; done
  for f in $FILES; do git diff --no-index -- /dev/null $f; done > $RUNLOG/new_files_$STAMP.patch
  print "runbook sha256 $(sha audit/dataset/tools/d3_gohr_runbook.zsh); new-file diff sha256 $(sha $RUNLOG/new_files_$STAMP.patch)"
  mark check "$(files_state)"
  ;;
commit)
  require content "$(sha $PREREG)"
  require tests "$(files_state)"
  require check "$(files_state)"
  [[ -z "$(git diff --cached --name-only)" ]] || die "the index is not empty"
  print "Will stage exactly:"; print -l -- $FILES
  print "Author: $(git var GIT_AUTHOR_IDENT)"
  authorize "COMMIT GOHR D3 PREREGISTRATION"
  git add -- $FILES || die "git add failed"
  [[ "$(git diff --cached --name-only | sort)" == "$(print -l -- $FILES | sort)" ]] || die "staged set differs from the 15 files; unstage ONLY those paths: git restore --staged -- <path>"
  run commit git commit -m "Preregister Gohr D3 (Speck32/64, 5 rounds) with instance-bound D0, D1 and D3 tooling"
  ;;
anchor)
  post_commit_ids
  [[ "$(git show HEAD:$PREREG | shasum -a 256 | cut -d' ' -f1)" == "$ANCHOR" ]] || die "HEAD blob differs from the preregistration file"
  run anchor $PY -B -c "import audit.dataset.d3_distribution_statistics as d3; b=d3.verify_preregistration('$PREREG', expected_sha256='$ANCHOR'); print('PREREG VERIFIED', b.commit, b.commit_utc)"
  print "ANCHOR=$ANCHOR PC=$PC"
  mark anchor "$PC $ANCHOR"
  ;;
calibrate)
  post_commit_ids; require anchor "$PC $ANCHOR"; fresh $CAL
  authorize "RUN GOHR CALIBRATION $PC"
  run calibrate /usr/bin/time -p $PY -B -m audit.dataset.tools.d3_gohr_method_validity_calibration --out-dir $CAL
  grep -q '"status": "IMPLEMENTATION_CONSISTENT"' $CAL/d3_method_validity_run_manifest.json || die "calibration status is not IMPLEMENTATION_CONSISTENT"
  mark calibrate "$PC $(sha $CAL/d3_method_validity.json)"
  ;;
calverify)
  post_commit_ids; require calibrate "$PC $(sha $CAL/d3_method_validity.json)"
  run calverify $PY -B -c "
import json, subprocess, hashlib, audit.dataset.d3_distribution_statistics as d3
from audit.dataset.experiments import d3_gohr_confirmatory as R
cal = d3.verify_calibration_evidence('$CAL', tool_path=R.CALIBRATION_TOOL_PATH, test_suite_path=R.CALIBRATION_SUITE_PATH)
R.check_protocol(d3.validate_preregistration_content(json.load(open('$PREREG'))), cal.design())
h = lambda p: hashlib.sha256(subprocess.run(['git', 'show', 'HEAD:' + p], capture_output=True, check=True).stdout).hexdigest()
assert cal.engine_sha256 == h('audit/dataset/d3_distribution_statistics.py') == '6d1ebe9e1857318642f1a022d18289078f5eaeefb3a5d9268e9d9b36275bfff7'
assert cal.tool_sha256 == h('audit/dataset/tools/d3_gohr_method_validity_calibration.py') == '5489325fa312994fd6f67e10dc06ec9223d192fcd04b3d12ffb23274e7d38e8d'
assert cal.test_suite_sha256 == h('audit/dataset/tests/test_d3_method_validity.py') == '808dedc1aeb0b32ab8dc30fc72710e5c867bb01f2c241be36a931df300a29d32'
print('CALIBRATION VERIFIED run', cal.run_id, 'evidence', cal.evidence_sha256)"
  mark calverify "$PC $(sha $CAL/d3_method_validity.json)"
  ;;
d0)
  post_commit_ids; require calverify "$PC $(sha $CAL/d3_method_validity.json)"; fresh $D0
  [[ -z "$(git status --porcelain -- speck.py audit/dataset/canonical.py audit/dataset/adapters/gohr.py audit/dataset/tools/d0_gohr_attested_build.py)" ]] || die "a pinned source is modified"
  authorize "BUILD GOHR D0 $PC"
  run d0_build /usr/bin/time -p $PY -B -m audit.dataset.tools.d0_gohr_attested_build build --out-dir $D0
  run d0_attestation $PY -B -m audit.dataset.tools.d0_gohr_attested_build verify --out-dir $D0
  run d0_canonical_verify $PY -B -m audit.dataset.canonical verify --out $D0
  shasum -a 256 $D0/* | tee $RUNLOG/d0_$STAMP.sha || die "hashing D0 failed"
  mark d0 "$PC $(sha $D0/d0_gohr_build_attestation.json)"
  ;;
d1)
  post_commit_ids; require d0 "$PC $(sha $D0/d0_gohr_build_attestation.json)"; fresh $D1OUT
  authorize "RUN GOHR D1 $PC"
  run d1 $PY -B -m audit.dataset.experiments.d1_gohr_canonical_census --instance-dir $D0 --out-dir $D1OUT
  run d1_gate $PY -B -c "
from audit.dataset.experiments import d3_gohr_confirmatory as R
print('D1 GATE PASSED', R.verify_d1_evidence('$D1OUT', '$D0'))"
  mark d1 "$PC $(sha $D1OUT/d1_gohr_certificate.json)"
  ;;
preflight)
  post_commit_ids
  require calverify "$PC $(sha $CAL/d3_method_validity.json)"
  require d1 "$PC $(sha $D1OUT/d1_gohr_certificate.json)"
  fresh $D3OUT
  run preflight $PY -B -m audit.dataset.experiments.d3_gohr_confirmatory --instance-dir $D0 --d1-dir $D1OUT --out-dir $D3OUT --preregistration $PREREG --preregistration-sha256 $ANCHOR --calibration-dir $CAL --preflight-only
  [[ ! -e $D3OUT ]] || die "preflight created $D3OUT"
  print "PREFLIGHT WROTE NOTHING"
  mark preflight "$PC $ANCHOR $(sha $CAL/d3_method_validity.json) $(sha $D0/d0_gohr_build_attestation.json) $(sha $D1OUT/d1_gohr_certificate.json)"
  ;;
d3)
  post_commit_ids
  require preflight "$PC $ANCHOR $(sha $CAL/d3_method_validity.json) $(sha $D0/d0_gohr_build_attestation.json) $(sha $D1OUT/d1_gohr_certificate.json)"
  fresh $D3OUT
  authorize "RUN GOHR D3 PRODUCTION $PC"
  run d3 $PY -B -m audit.dataset.experiments.d3_gohr_confirmatory --instance-dir $D0 --d1-dir $D1OUT --out-dir $D3OUT --preregistration $PREREG --preregistration-sha256 $ANCHOR --calibration-dir $CAL
  mark d3 "$PC $(sha $D3OUT/d3_gohr_sealed_audit.json)"
  ;;
verify)
  post_commit_ids; require d3 "$PC $(sha $D3OUT/d3_gohr_sealed_audit.json)"
  run verify $PY -B -m audit.dataset.tools.d3_gohr_postrun_verify $D3OUT $D0 $D1OUT $PREREG $ANCHOR $CAL
  shasum -a 256 $PREREG $CAL/* $D0/* $D1OUT/* $D3OUT/* | tee $RUNLOG/final_inventory_$STAMP.sha || die "inventory failed"
  ;;
*)
  die "unknown stage '$STAGE'"
  ;;
esac
print "STAGE $STAGE COMPLETE"
