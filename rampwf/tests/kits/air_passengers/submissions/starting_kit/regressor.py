from sklearn.ensemble import RandomForestRegressor


def get_component():
    regressor = RandomForestRegressor(
        n_estimators=10, max_depth=10, max_features=10
    )
    return regressor
