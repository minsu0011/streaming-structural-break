"""Exact offline explanation flags for fixed synthetic viewer paths only."""
import numpy as np

STATUS_NAMES = ('H_UNSUPPORTED', 'FOUR_BLOCK_WARMUP', 'SCALE_UNCONFIRMED',
                'SCALE_HARD_CEILING', 'SCALE_ADMISSION_REJECTED', 'SCALE_ADMISSION_PASSED')


def evidence_flags(path, scale_cut):
    values = np.asarray(path)
    if values.ndim != 2 or values.shape[1] != 45 or not np.isfinite(values).all() or not scale_cut > 0:
        raise ValueError('Finite fixed G2 path and original positive scale cutoff required')
    n = len(values)
    flags = np.zeros((n, 10), dtype=np.int64)
    for t in range(n):
        row = values[t]
        complete = int(row[16])
        last = 16 * complete - 1
        supported = bool(row[43] > .5)
        if not supported:
            status = 0
        elif complete < 4:
            status = 1
        elif row[11] <= 1:
            status = 2
        elif row[42] >= 4 / scale_cut:
            status = 3
        elif row[11] <= row[42]:
            status = 4
        else:
            status = 5
        z = values[np.arange(max(0, complete - 4), complete) * 16 + 15, 9]
        assert np.all(np.abs(z) <= 4)
        core_pass = supported and complete >= 4 and row[13] > row[42]
        scale_pass = status == 5
        if last >= 0:
            assert bool(values[last, 40] > .5) == bool(core_pass or scale_pass)
        flags[t] = (status, last, int(row[15] > .5), int((z < 0).sum()), int((z > 0).sum()),
                    int(core_pass), int(scale_pass), int(row[40] > .5), int(row[30] > 0), int(len(z)))
    return flags


FLAG_COLUMNS = ('scale_status', 'last_complete_point', 'new_block_now', 'negative_blocks',
                'positive_blocks', 'core_pass_on_latest_block', 'scale_pass_on_latest_block',
                'qualified_fresh_now', 'previous_or_current_qualified_memory', 'available_recent_blocks')
