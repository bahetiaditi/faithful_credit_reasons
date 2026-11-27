import pytest

from fcr.data.naming import TableName, feature_suffix, parse_table_name


@pytest.mark.parametrize(
    ("stem", "expected"),
    [
        ("train_base", TableName("train", "base", 0, None)),
        ("test_base", TableName("test", "base", 0, None)),
        ("train_static_0_0", TableName("train", "static", 0, 0)),
        ("train_static_cb_0", TableName("train", "static_cb", 0, None)),
        ("train_person_1", TableName("train", "person", 1, None)),
        ("train_person_2", TableName("train", "person", 2, None)),
        ("train_applprev_1_1", TableName("train", "applprev", 1, 1)),
        ("train_credit_bureau_a_2_10", TableName("train", "credit_bureau_a", 2, 10)),
        ("train_credit_bureau_b_1", TableName("train", "credit_bureau_b", 1, None)),
        ("train_tax_registry_a_1", TableName("train", "tax_registry_a", 1, None)),
        ("test_debitcard_1", TableName("test", "debitcard", 1, None)),
    ],
)
def test_parse_real_table_names(stem, expected):
    parsed = parse_table_name(stem)
    assert parsed == expected
    assert parsed.stem() == stem


@pytest.mark.parametrize("stem", ["train_static", "valid_base", "train_static_3", "feature_definitions"])
def test_reject_non_table_names(stem):
    with pytest.raises(ValueError):
        parse_table_name(stem)


@pytest.mark.parametrize(
    ("column", "suffix"),
    [
        ("actualdpd_943P", "P"),
        ("annuity_853A", "A"),
        ("birth_259D", "D"),
        ("education_927M", "M"),
        ("riskassesment_940T", "T"),
        ("byoccupationinc_3656910L", "L"),
        ("case_id", None),
        ("num_group2", None),
        ("WEEK_NUM", None),
    ],
)
def test_feature_suffix(column, suffix):
    assert feature_suffix(column) == suffix


@pytest.mark.parametrize("column", ["income", "income_12X", "income_A"])
def test_feature_suffix_rejects_unknown_columns(column):
    with pytest.raises(ValueError):
        feature_suffix(column)
