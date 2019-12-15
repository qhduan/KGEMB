
import tensorflow as tf
from entity_id import EntityID


def build_model(n_ent_rel, embedding_size=64):
    input_real = tf.keras.Input(shape=(None,), dtype='string')
    input_fake = tf.keras.Input(shape=(None,), dtype='string')

    # e2i 用来把一个字符串转换为int
    # e2i 是一个 Model 是因为我们要单独存
    e2i = EntityID(n_ent_rel)
    # emb 用来把一个 int 转换为一个 float vector
    # emb 是一个 Model 是因为我们要单独存
    emb = tf.keras.Sequential([
        tf.keras.Input(shape=(None,), dtype=tf.float32),
        tf.keras.layers.Embedding(n_ent_rel, embedding_size)
    ])

    real = e2i(input_real)
    fake = e2i(input_fake)

    inputs = [
        real[:, 0],
        real[:, 1],
        real[:, 2],
        fake[:, 0],
        fake[:, 1],
        fake[:, 2],
    ]

    (
        real_head,
        real_rel,
        real_tail,
        fake_head,
        fake_rel,
        fake_tail
    ) = [
        emb(x)
        for x in inputs
    ]

    dis_real = tf.linalg.normalize(real_head + real_rel - real_tail, axis=1)[1]
    dis_fake = tf.linalg.normalize(fake_head + fake_rel - fake_tail, axis=1)[1]

    x = dis_real - dis_fake

    model = tf.keras.models.Model(
        inputs=[input_real, input_fake],
        outputs=x
    )

    model.compile(
        optimizer='adam',
        # 自定义loss函数
        # loss(x, y) = max(0, -y * (x1 - x2) + margin)
        # y = -1
        # margin = 1.0
        loss=lambda true, pred: tf.math.maximum(0.0, pred + true)
    )
    return model, e2i, emb
