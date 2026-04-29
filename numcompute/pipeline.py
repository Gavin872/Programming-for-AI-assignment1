class Pipeline:
    def __init__(self, steps):
        if not isinstance(steps, list) or len(steps) == 0:
            raise ValueError("steps must be a non-empty list.")
        self.steps = steps

    def fit(self, X, y=None):
        Xt = X
        for i, (_, step) in enumerate(self.steps):
            is_last = i == len(self.steps) - 1
            if is_last and hasattr(step, "predict") and hasattr(step, "fit"):
                step.fit(Xt, y)
            else:
                if hasattr(step, "fit_transform"):
                    Xt = step.fit_transform(Xt)
                elif hasattr(step, "fit") and hasattr(step, "transform"):
                    step.fit(Xt)
                    Xt = step.transform(Xt)
                else:
                    raise ValueError("Each transformer step must implement fit/transform.")
        return self

    def transform(self, X):
        Xt = X
        for _, step in self.steps:
            if hasattr(step, "transform"):
                Xt = step.transform(Xt)
            else:
                break
        return Xt

    def fit_transform(self, X, y=None):
        Xt = X
        for i, (_, step) in enumerate(self.steps):
            is_last = i == len(self.steps) - 1
            if is_last and hasattr(step, "predict") and hasattr(step, "fit") and not hasattr(step, "transform"):
                step.fit(Xt, y)
            else:
                if hasattr(step, "fit_transform"):
                    Xt = step.fit_transform(Xt)
                elif hasattr(step, "fit") and hasattr(step, "transform"):
                    step.fit(Xt)
                    Xt = step.transform(Xt)
                else:
                    raise ValueError("Each transformer step must implement fit/transform.")
        return Xt

    def predict(self, X):
        Xt = X
        for i, (_, step) in enumerate(self.steps):
            is_last = i == len(self.steps) - 1
            if is_last and hasattr(step, "predict"):
                return step.predict(Xt)
            if hasattr(step, "transform"):
                Xt = step.transform(Xt)
            else:
                raise ValueError("Intermediate steps must implement transform.")
        raise ValueError("Last step does not implement predict.")