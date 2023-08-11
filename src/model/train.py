from argparse import ArgumentParser
from pathlib import Path


from numpy import array
from pandas import read_csv, concat, DataFrame
from mlflow import autolog
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


__args__ = {
    "--training_data": {
        "dest": "training_data",
        "type": str,
        "default": "production/data",
    },
    "--reg_rate": {
        "dest": "reg_rate",
        "type": float,
        "default": 0.01,
    },
    "--random_state": {
        "dest": "random_state",
        "type": int,
        "default": 0,
    },
    "--test_size": {
        "dest": "test_size",
        "type": float,
        "default": 0.30,
    },
}


def parse_args(
        parser: ArgumentParser = ArgumentParser(),
        args: dict = __args__,
) -> ArgumentParser:
    for key in args:
        parser.add_argument(key, **args[key])
    return parser.parse_args()


def get_csv_df(path: Path) -> DataFrame:
    print(path)
    if not isinstance(path, Path):
        path = Path(path)
    if not path.exists():
        raise FileExistsError(f"Cannot use non-existent path provided: {path}")
    elif not (csv_files := list(path.rglob("*.csv"))):
        raise RuntimeError(f"No CSV files found in provided data path: {path}")
    elif (nfiles := len(csv_files)) == 0:
        msg = f"Multiple CSV files found in provided data path: {path}"
        raise RuntimeError(msg)
    elif nfiles == 1:
        return read_csv(csv_files[0])
    return concat((read_csv(f) for f in csv_files), sort=False)


def split_data(
        df: DataFrame,
        args: ArgumentParser,
        xcols: list = [
            "Pregnancies",
            "PlasmaGlucose",
            "DiastolicBloodPressure",
            "TricepsThickness",
            "SerumInsulin",
            "BMI",
            "DiabetesPedigree",
            "Age"
        ],
        ycol: str = "Diabetic",
) -> tuple:
    X, y = df.loc[:, xcols].values, df.loc[:, ycol].values
    return train_test_split(
        X,
        y,
        random_state=args.random_state,
        test_size=args.test_size,
    )


def train_model(
        reg_rate: float,
        X_train: array,
        y_train: array,
        solver: str = "liblinear",
) -> LogisticRegression:
    C = 1 / reg_rate
    return LogisticRegression(
        C=C,
        solver=solver
    ).fit(X_train, y_train)


def main(
        args: ArgumentParser = parse_args(),
) -> LogisticRegression.decision_function:
    print("starting log")
    autolog()
    # read data
    print("reading csv")
    df = get_csv_df(args.training_data)

    # split data
    print("splitting data")
    X_train, X_test, y_train, y_test = split_data(
        df,
        args,
    )
    print("training model")
    res = train_model(
        args.reg_rate,
        X_train,
        y_train
    )
    return res, X_test, y_test


if __name__ == "__main__":
    # add space in logs
    print("\n\n")
    print("*" * 60)

    # run main function
    main()

    # add space in logs
    print("*" * 60)
    print("\n\n")
