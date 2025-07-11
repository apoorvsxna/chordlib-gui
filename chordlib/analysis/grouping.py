import numpy as np
from typing import List, Dict, Any


def group_chord_events(
    times: np.ndarray,
    chords: List[str],
    strengths: List[float],
    min_duration_sec: float = 0.1,
) -> List[Dict[str, Any]]:
    """
    group consecutive identical chords into events with start/end times.
    get rid of very short/abrupt events (<min_duration_sec).
    """
    events: List[Dict[str, Any]] = []
    if not chords:
        return events

    current_label = chords[0]
    current_start = float(times[0])
    accum_strengths = [strengths[0]]

    for idx in range(1, len(chords)):
        label = chords[idx]
        t = float(times[idx])
        s = strengths[idx]

        if label == current_label:
            accum_strengths.append(s)
        else:
            duration = t - current_start
            if duration >= min_duration_sec and current_label != "N":
                events.append(
                    {
                        "start": current_start,
                        "end": t,
                        "chord": current_label,
                        "strength": float(np.mean(accum_strengths)),
                    }
                )
            current_label = label
            current_start = t
            accum_strengths = [s]

    if len(times) >= 2:
        last_interval = float(times[-1] - times[-2])
    else:
        last_interval = min_duration_sec
    final_end = float(times[-1]) + last_interval
    final_dur = final_end - current_start
    if final_dur >= min_duration_sec and current_label != "N":
        events.append(
            {
                "start": current_start,
                "end": final_end,
                "chord": current_label,
                "strength": float(np.mean(accum_strengths)),
            }
        )

    return events
