from pandas import MultiIndex
from pandas.core.groupby import DataFrameGroupBy

from common import read_source_csv, get_selected_data_csv_file_path


def _select_by_significance(grouped_data: DataFrameGroupBy, col_name: str = "Spend",
                            significance_thresh: float = 0.9,
                            ) -> MultiIndex:
    agg_spends = grouped_data[col_name].sum().sort_values(ascending=False)

    # Select only materials with positive spends
    agg_spends = agg_spends[agg_spends > 0]

    # Select only materials with the largest spends that make up `significance_thresh` of the total spend
    agg_spends = agg_spends[agg_spends.cumsum() <= agg_spends.sum() * significance_thresh]

    return agg_spends.index


def _select_by_frequency(grouped_data: DataFrameGroupBy, frequency_thresh: float = 0.4) -> MultiIndex:
    grouped_epoch_months = grouped_data["EpochM"]

    record_counts = grouped_epoch_months.count()
    frequencies = record_counts / (grouped_epoch_months.max() - grouped_epoch_months.min() + 1)

    # Select only materials with more than one transactions
    frequencies = frequencies[record_counts > 1]

    # Select only materials with higher transaction frequency than `frequency_thresh`
    frequencies = frequencies[frequencies >= frequency_thresh]

    return frequencies.index


def main(company_name: str):
    data = read_source_csv(company_name=company_name)

    grouped_data = data.groupby(["MaterialNo", "MaterialGroupNo"])

    selected_significance_inds = _select_by_significance(grouped_data=grouped_data)
    selected_frequency_inds = _select_by_frequency(grouped_data=grouped_data)

    selected_inds = selected_significance_inds.intersection(selected_frequency_inds)
    selected_data = data.set_index(["MaterialNo", "MaterialGroupNo"]).loc[selected_inds]

    print(
        f"company_name =                                                                                        "
        f"{company_name}"
    )
    print(
        f"total number of rows =                                                                                "
        f"{len(data)}"
    )
    print(
        f"number of unique (\"MaterialNo\", \"MaterialGroupNo\") pairs =                                        "
        f"{len(grouped_data)}"
    )
    print(
        f"number of unique (\"MaterialNo\", \"MaterialGroupNo\") pairs selected by significance =               "
        f"{len(selected_significance_inds)}"
    )
    print(
        f"number of unique (\"MaterialNo\", \"MaterialGroupNo\") pairs selected by frequency =                  "
        f"{len(selected_frequency_inds)}"
    )
    print(
        f"number of unique (\"MaterialNo\", \"MaterialGroupNo\") pairs selected by significance and frequency = "
        f"{len(selected_inds)}"
    )
    print(
        f"total number of selected rows =                                                                       "
        f"{len(selected_data)}"
    )
    print()

    selected_data.to_csv(get_selected_data_csv_file_path(company_name=company_name))


if __name__ == '__main__':
    # TODO: add dry run mode
    # TODO: argpase
    main(company_name="A")
    main(company_name="B")
    main(company_name="C")
