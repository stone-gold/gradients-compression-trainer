"""Trainer entrypoint: dispatches on --task-type."""

import logging
import sys
import traceback

from quant import common


log = logging.getLogger("quant.main")


def main(argv=None) -> int:
    common.setup_logging()
    args = common.parse_args(argv)
    log.info("task %s type %s model %s hours %s", args.task_id, args.task_type, args.model, args.hours_to_complete)
    if args.task_type == "QuantFixedTask":
        from quant import fixed

        fixed.run(args)
    elif args.task_type == "QuantFreeformTask":
        from quant import freeform  # added by the freeform track

        freeform.run(args)
    else:
        log.error("unsupported task type %s", args.task_type)
        return 1
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        # A valid manifest may already be in place: exit 0 only if it is, so the best checkpoint so far is uploaded.
        import os

        args = common.parse_args()
        out = common.output_dir(args.task_id, args.expected_repo_name)
        sys.exit(0 if (out / common.MANIFEST).exists() else 1)
