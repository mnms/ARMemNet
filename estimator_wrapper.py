import tensorflow as tf

class EstimatorWrapper(object):
    """
    Simple wrapper around an estimator that is coming from a file.
    We override __getstate__() and __setstate__() to allow easy serialization
    of this instance on the workers when this is used in a UDF.
    """
    def __init__(self, model_path: str):
        self.estimator = tf.saved_model.load(model_path)
        self.model_path = model_path

    def __getstate__(self):
        return self.model_path

    def __setstate__(self, model_path: str):
        self.estimator = tf.saved_model.load(model_path)
        self.model_path = model_path
