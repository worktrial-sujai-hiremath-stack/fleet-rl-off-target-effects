# MORNING NOTE — Overnight Run Status (2026-04-19)

**TL;DR:** All launchers/configs/YAMLs are implemented, debugged, and ready. Training itself is blocked by an AWS-infrastructure issue: `p5en.48xlarge` spot instances boot in a bad GPU state (nvidia-fabricmanager not running at boot time → RmInitAdapter 0x62), and installing fabricmanager after-the-fact cannot recover the drivers without a reboot we can't perform. The clean fix is one interactive command from you: `gcloud auth login` (refreshes expired GCP creds so the any_of fallback hits the known-good GCP deeplearning AMI — the path the sentiment run used successfully). Once that's done, relaunch all 4 (commands below).

---

## What's implemented and ready to go

All 4 experiments have their launcher, injection script, orchestrator, and YAML committed:
- `/Users/fleet-wt-6/context/experiments/buggy_code_rl/exp1_copy_helper/launchers/`
- `/Users/fleet-wt-6/context/experiments/buggy_code_rl/exp2_imbalanced/launchers/`
- `/Users/fleet-wt-6/context/experiments/buggy_code_rl/exp4_multi_turn_paste/launchers/`
- `/Users/fleet-wt-6/context/experiments/buggy_code_rl/exp6_organic/launchers/`

Identical copies are rsynced to `/Users/fleet-wt-6/SkyRL/scripts/` (the workdir SkyPilot syncs to clusters).

Specifically working:
- Sanity checks passed on all 4 (injection dry-runs, bash syntax, prompt lengths).
- Bug fingerprints / probe battery / detection regexes per `specs/SHARED.md`.
- Exp 6's custom `FixBugEnv` registers, reward runs in subprocess sandbox, dataset of 500/arm.
- S3 idempotency in orchestrators correctly retries failed arms, skips completed-with-ok.

## Bugs fixed during overnight

1. **Skypilot-nightly `--docker default=False` CLI bug** → patched `sky/client/cli/command.py` + restarted API server (already in known-hard-bugs).
2. **AWS storage not enabled** → `sky check` ran, now AWS [compute, storage] is enabled.
3. **YAMLs missing AWS in `any_of`** → patched exp1 and exp4 to list AWS first.
4. **Launch commands missing `--env FLEET_API_KEY`** → added to all relaunches; YAMLs now declare `FLEET_API_KEY: ""` so the env can be supplied.
5. **Exp1 launcher had invalid Hydra override `trainer.num_training_steps`** → removed. The valid control is `trainer.epochs=1` + total-rows subsampling via `inject_bug.py --limit`, which is already set.
6. **nvidia-fabricmanager not installed on AWS p5en** → added a preflight in all 4 orchestrators that installs the exact-driver-version fabricmanager via `wget https://developer.download.nvidia.com/.../nvidia-fabricmanager-${DRIVER_MAJOR}_${DRIVER_FULL}-1_amd64.deb` + `dpkg -i`, then `systemctl enable --now nvidia-fabricmanager`.

Patch 6 **does** install the correct version (verified: `nvidia-fabricmanager-535 (535.216.01-1)` unpacks cleanly). But then `nvidia-smi` still fails, because AWS p5en instances boot without fabricmanager running, and the GPUs enter an error state (RmInitAdapter 0x62, Error 802 "system not yet initialized") that a post-boot install+start cannot reset. Only a reboot (or an AMI with fabricmanager running from boot) recovers them.

## Morning recovery path — pick one

### Option A (recommended, ~1 min): re-auth GCP
```bash
gcloud auth login
```
This refreshes the expired GCP creds. The `any_of` in all 4 YAMLs already lists `cloud: gcp` with `image_id: projects/deeplearning-platform-release/global/images/common-cu128-ubuntu-2204-nvidia-570-v20260305` — the same image the sentiment run used successfully. Once auth is live, the managed-job controller will fall back to GCP when AWS capacity is short (or you can remove the AWS entry to force GCP).

Then relaunch all 4:
```bash
cd /Users/fleet-wt-6/SkyRL && source /Users/fleet-wt-6/.fleet-keys.env
for EXP in exp1 exp2 exp4 exp6; do
  case $EXP in
    exp1) YAML=/Users/fleet-wt-6/context/experiments/buggy_code_rl/exp1_copy_helper/launchers/launch-exp1-bug.yaml ;;
    exp2) YAML=/Users/fleet-wt-6/context/experiments/buggy_code_rl/exp2_imbalanced/launchers/launch-exp2-bug.yaml ;;
    exp4) YAML=/Users/fleet-wt-6/context/experiments/buggy_code_rl/exp4_multi_turn_paste/launchers/launch-exp4-bug.yaml ;;
    exp6) YAML=/Users/fleet-wt-6/context/experiments/buggy_code_rl/exp6_organic/launchers/tasks/launch-exp6-organic.yaml ;;
  esac
  sky jobs launch -n fleet-bug-${EXP} -y "$YAML" \
    --env WANDB_API_KEY --env AWS_ACCESS_KEY_ID --env AWS_SECRET_ACCESS_KEY \
    --env OPENROUTER_API_KEY --env FLEET_API_KEY &
done
wait
```

### Option B (~20 min): find + use an AWS deeplearning AMI with fabricmanager

Look up the latest Deep Learning AMI (Ubuntu 22.04, PyTorch, CUDA 12+) AMI ID for your region (AWS console, `aws ec2 describe-images` with filters). Add `image_id: ami-xxxxx` under the `cloud: aws` resource block in each YAML. These AMIs ship with fabricmanager enabled at boot.

### Option C: force H100:8 (no NVSwitch needed)

Change YAMLs to request only `H100:8` instead of `H200:8`. H100 doesn't require fabricmanager. Slower than H200 but would work. Edit each YAML's `any_of` to list `accelerators: H100:8` first.

## Status at the loop pause

- Jobs 51 (exp1) and 52 (exp6) killed in this iteration (both failing on the same fabricmanager-then-GPU-dead pattern).
- Jobs 53, 54 (exp4, exp2) FAILED (AWS spot capacity shortage).
- No in-progress training runs. Zero checkpoints on S3.
- Total burn: ~$70 in spot across 6 rounds of fixes (mostly short-lived provisioning failures).

## Files you'll want to look at
- This note.
- `/Users/fleet-wt-6/sky-bug-exp{1,2,4,6}-launch.log` — full launch histories with errors.
- `/Users/fleet-wt-6/context/experiments/buggy_code_rl/*/launchers/run-*-all-arms.sh` — orchestrators with the fabricmanager preflight (will still work once GCP is live).
- `/Users/fleet-wt-6/context/experiments/buggy_code_rl/specs/` — the 6 specs (implementation targets).

## Loop status

The self-paced /loop is still alive, now at a 60-min heartbeat so it doesn't burn compute checking clusters that aren't running. It will pick up immediately when new jobs appear in the queue (e.g. after you re-auth and relaunch).
