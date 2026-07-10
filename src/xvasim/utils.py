import typing

import numpy as np


def dates_to_years(
    dates: typing.Iterable[typing.Any],
    valuation_date: typing.Any,
) -> np.ndarray:
    """Convert an array of dates into time in years relative to a valuation date.

    Args:
        dates: An iterable (list, tuple, or numpy array) of dates. Dates can be
            datetime.date, datetime.datetime, numpy.datetime64, or ISO strings.
        valuation_date: The reference valuation date. Can be datetime.date,
            datetime.datetime, numpy.datetime64, or an ISO string.

    Returns:
        A 1D numpy array of floats representing the time points in years,
        using 365.25 days per year.
    """
    # Convert valuation date to numpy datetime64 with day precision
    val_dt = np.datetime64(valuation_date, "D")

    # Convert input dates to numpy datetime64 array with day precision
    dates_arr = np.array(dates, dtype="datetime64[D]")

    # Calculate difference in days and convert to float
    diff_days = (dates_arr - val_dt).astype(float)

    # Convert days to years (using 365.25 days per year)
    return diff_days / 365.25
