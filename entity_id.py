
import tensorflow as tf


class EntityID(tf.keras.models.Model):
    """用来做实体到某个具体int的转换"""

    def __init__(self, num_buckets=10*10000):
        super(EntityID, self).__init__()
        self.num_buckets = num_buckets

    def call(self, inputs):
        """利用 to_hash_bucket_fast 来快速构建词表"""
        x = tf.strings.to_hash_bucket_fast(inputs, self.num_buckets)
        x = tf.cast(x, dtype=tf.float32)
        return x
