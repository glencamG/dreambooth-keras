import tensorflow as tf

class DDIMNoiseScheduler(tf.keras.callbacks.Callback):
    def __init__(self, noise_schedule):
        super().__init__()
        self.noise_schedule = noise_schedule

    def on_train_batch_begin(self, batch, logs=None):
        timesteps = tf.random.uniform(
            shape=(self.params["batch_size"],),
            minval=0,
            maxval=self.noise_schedule.train_timesteps,
            dtype=tf.int32,
        )
        noise_scale = self.noise_schedule.scale(timesteps)
        noise = tf.random.normal(
            shape=(self.params["batch_size"], self.noise_schedule.noise_dim),
            dtype=tf.float32,
        )
        noise *= tf.reshape(noise_scale, (-1, 1))
        self.model.layers[1].diffusion_model.noise = noise
