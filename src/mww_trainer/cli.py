"""`mww` — command-line entry point for the microWakeWord trainer.

Each subcommand lives in its own module and is imported only when used, so
`mww --help` and the light tools never pay for importing TensorFlow.
"""

import importlib
import sys

# name -> (module, one-line help)
COMMANDS = {
    "setup": ("mww_trainer.cli", "Download the Piper binary for this machine and verify the environment"),
    "run": ("mww_trainer.pipeline", "Run training pipeline phases from a YAML config"),
    "test": ("mww_trainer.evaluate", "Run a TFLite model over WAV files, pymicro_wakeword-style"),
    "batch-test": ("mww_trainer.batch_test", "Summarise which false-positive clips still trigger"),
    "ab-test": ("mww_trainer.ab_test", "Compare two models on the same false-positive clips"),
    "elevenlabs": ("mww_trainer.elevenlabs", "Generate extra positives with the ElevenLabs API"),
    "ha-tts": ("mww_trainer.ha_tts", "Generate clips from Home Assistant's TTS engines"),
    "channel-probe": ("mww_trainer.channel_probe", "Put clips through the room/noise channel for inspection"),
}


def _usage() -> str:
    width = max(len(n) for n in COMMANDS)
    lines = [
        "usage: mww <command> [options]",
        "",
        "microWakeWord trainer. Run `mww <command> --help` for a command's options.",
        "",
        "commands:",
    ]
    lines += [f"  {name:<{width}}  {help_}" for name, (_, help_) in COMMANDS.items()]
    lines += [
        "",
        "typical run:",
        "  mww setup",
        "  mww run --config my_wakeword.yaml",
        "  mww test --model my_wakeword_mww.tflite record/*.wav",
    ]
    return "\n".join(lines)


def setup_main(argv=None, prog=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog=prog,
        description="Download the Piper binary for this CPU architecture into "
                    "training/ and check that every training dependency imports. "
                    "Safe to re-run.",
    )
    parser.parse_args(argv)

    from mww_trainer import pipeline

    pipeline.WORK_DIR.mkdir(parents=True, exist_ok=True)
    pipeline.phase_setup()
    return 0


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(_usage())
        return 0

    name, rest = argv[0], argv[1:]
    if name not in COMMANDS:
        print(f"mww: unknown command {name!r}\n", file=sys.stderr)
        print(_usage(), file=sys.stderr)
        return 2

    prog = f"mww {name}"
    if name == "setup":
        return setup_main(rest, prog)
    module = importlib.import_module(COMMANDS[name][0])
    return module.main(rest, prog=prog) or 0


if __name__ == "__main__":
    sys.exit(main())
