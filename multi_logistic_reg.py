import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


def read_data(file) -> pd.DataFrame:
    data = pd.read_csv(file)
    print(data.columns)
    return data


def divide_DLIT_AG(df: pd.DataFrame) -> pd.DataFrame:
    for i in range(8):
        # Create a new column for each value of DLIT_AG, set to 1 where
        # DLIT_AG equals i, else 0
        df[f"DLIT_AG{i}"] = (df['DLIT_AG'] == i).astype(int)
    # Drop the original DLIT_AG column
    df.drop('DLIT_AG', axis=1, inplace=True)
    return df


def divide_by_age(df: pd.DataFrame) -> dict[pd.DataFrame]:
    age_groups = {
        '<50': df[df['AGE'] < 50],
        '50-60': df[(df['AGE'] >= 50) & (df['AGE'] <= 60)],
        '60-70': df[(df['AGE'] > 60) & (df['AGE'] <= 70)],
        '70-80': df[(df['AGE'] > 70) & (df['AGE'] <= 80)],
        '>80': df[(df['AGE'] > 80)]
    }
    return age_groups


def get_variables(df: pd.DataFrame):
    y = np.asarray(df["Status"])
    X = np.asarray(df[['AGE', 'L_BLOOD', 'endocr_01',
                       'endocr_02', 'S_AD_KBRIG',
                       'B_BLOK_S_n', 'NA_KB', 'DLIT_AG0',
                       'DLIT_AG1', 'DLIT_AG2',
                       'DLIT_AG3', 'DLIT_AG4',
                       'DLIT_AG5', 'DLIT_AG6', 'DLIT_AG7']]
                   )
    return X, y


def logistic_reg(X, y) -> tuple:
    model = LogisticRegression()
    mod = model.fit(X, y)
    # Coefficients and Intercept
    coefficients = mod.coef_
    intercept = mod.intercept_
    return intercept, coefficients


def main():
    data = read_data("selected_df.csv")
    data = divide_DLIT_AG(data)
    age_group = divide_by_age(data)
    for age in age_group:
        X, y = get_variables(age_group[age])
        intercept, coefficients = logistic_reg(X, y)
        print(intercept, coefficients)
        age_group[age].to_csv(f"age_group_{age}.csv")
        

    # data.to_csv('logistic_data.csv')


if __name__ == "__main__":
    main()
