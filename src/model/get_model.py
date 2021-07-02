def get_model(cfg):
    if cfg["type"] == "box_reg":
        from .box_regression import BoundingBoxRegressor
        return BoundingBoxRegressor(cfg)
    else:
        raise NotImplementedError
