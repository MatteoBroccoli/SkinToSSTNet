import tensorflow as tf
from tensorflow import keras
from keras import layers

# define loss functions
LAMBDA = 100

loss_object = tf.keras.losses.BinaryCrossentropy(
    from_logits=True, reduction=tf.keras.losses.Reduction.NONE)


def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    total_gen_loss = gan_loss + (LAMBDA * l1_loss)
    return total_gen_loss, gan_loss, l1_loss

def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss

# build a GAN class
class GAN(keras.Model):
    def __init__(self, discriminator, generator):
        super().__init__()
        self.discriminator = discriminator
        self.generator = generator

    def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn
        self.d_loss_tracker = keras.metrics.Mean(name="d_loss")
        self.g_tot_loss_tracker = keras.metrics.Mean(name="g_tot_loss")
        self.gan_loss_tracker = keras.metrics.Mean(name="gan_loss")
        self.g_l1_loss_tracker = keras.metrics.Mean(name="g_l1_loss")

    def train_step(self, data):
        input_image, target = data
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = self.generator(input_image, training=True)

            disc_real_output = self.discriminator([input_image, target], training=True)
            disc_generated_output = self.discriminator([input_image, gen_output], training=True)

            gen_total_loss, gan_loss, g_l1_loss = self.g_loss_fn(disc_generated_output, gen_output, target)
            disc_loss = self.d_loss_fn(disc_real_output, disc_generated_output)

        generator_gradients = gen_tape.gradient(gen_total_loss,
                                                self.generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss,
                                                    self.discriminator.trainable_variables)

        self.g_optimizer.apply_gradients(zip(generator_gradients,
                                                self.generator.trainable_variables))
        self.d_optimizer.apply_gradients(zip(discriminator_gradients,
                                                    self.discriminator.trainable_variables))
        
        # Update metrics and return their value.
        self.d_loss_tracker.update_state(disc_loss)
        self.g_tot_loss_tracker.update_state(gen_total_loss)
        self.gan_loss_tracker.update_state(gan_loss)
        self.g_l1_loss_tracker.update_state(g_l1_loss)
        return {
                "d_loss": self.d_loss_tracker.result(),
                "g_tot_loss": self.g_tot_loss_tracker.result(),
                "gan_loss": self.gan_loss_tracker.result(),
                "g_l1_loss": self.g_l1_loss_tracker.result(),
            }
        
    def test_step(self, data):
        input_image, target = data

        gen_output = self.generator(input_image, training=False)
        disc_real_output = self.discriminator([input_image, target], training=False)
        disc_generated_output = self.discriminator([input_image, gen_output], training=False)

        gen_total_loss, gan_loss, g_l1_loss = self.g_loss_fn(disc_generated_output, gen_output, target)
        disc_loss = self.d_loss_fn(disc_real_output, disc_generated_output)

        # Update metrics and return their value.
        self.d_loss_tracker.update_state(disc_loss)
        self.g_tot_loss_tracker.update_state(gen_total_loss)
        self.gan_loss_tracker.update_state(gan_loss)
        self.g_l1_loss_tracker.update_state(g_l1_loss)
        return {
                "d_loss": self.d_loss_tracker.result(),
                "g_tot_loss": self.g_tot_loss_tracker.result(),
                "gan_loss": self.gan_loss_tracker.result(),
                "g_l1_loss": self.g_l1_loss_tracker.result(),
            }

# the generator is a U-Net
def get_generator():
    inputs = keras.Input(shape=(720,1440,5))
    # encoder (downsampler)
    initializer = tf.random_normal_initializer(0., 0.02)
    concat1 = layers.Conv2D(64, 4, 2, activation='relu', padding='same', kernel_initializer=initializer, use_bias=False)(inputs)
    initializer = tf.random_normal_initializer(0., 0.02)
    x = layers.Conv2D(128, 4, 2, activation='relu', padding='same', kernel_initializer=initializer, use_bias=False)(concat1)
    concat2 = layers.BatchNormalization()(x)
    initializer = tf.random_normal_initializer(0., 0.02)
    x = layers.Conv2D(256, 4, 2, activation='relu', padding='same', kernel_initializer=initializer, use_bias=False)(concat2)
    concat3 = layers.BatchNormalization()(x)
    initializer = tf.random_normal_initializer(0., 0.02)
    x = layers.Conv2D(512, 4, 2, activation='relu', padding='same', kernel_initializer=initializer, use_bias=False)(concat3)
    concat4 = layers.BatchNormalization()(x)
    initializer = tf.random_normal_initializer(0., 0.02)
    x = layers.Conv2D(512, 4, 2, activation='relu', padding='same', kernel_initializer=initializer, use_bias=False)(concat4)
    concat5 = layers.BatchNormalization()(x)
    initializer = tf.random_normal_initializer(0., 0.02)
    x = layers.Conv2D(512, 4, 2, activation='relu', padding='same', kernel_initializer=initializer, use_bias=False)(concat5)
    concat6 = layers.BatchNormalization()(x)
    initializer = tf.random_normal_initializer(0., 0.02)
    x = layers.Conv2D(512, 4, 2, activation='relu', padding='same', kernel_initializer=initializer, use_bias=False)(concat6)
    concat7 = layers.BatchNormalization()(x)
    initializer = tf.random_normal_initializer(0., 0.02)
    x = layers.Conv2D(512, 4, 2, activation='relu', padding='same', kernel_initializer=initializer, use_bias=False)(concat7)
    x = layers.BatchNormalization()(x)

    # decoder (upsampler)
    initializer = tf.random_normal_initializer(0., 0.02)
    x = layers.Conv2DTranspose(512, 4, 2, padding='same', kernel_initializer=initializer, activation='relu', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Concatenate()([x, concat7])
    initializer = tf.random_normal_initializer(0., 0.02)
    x = layers.Conv2DTranspose(512, 4, 2, padding='same', kernel_initializer=initializer, activation='relu', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Cropping2D(((0, 0), (0, 1)))(x)
    x = layers.Concatenate()([x, concat6])
    initializer = tf.random_normal_initializer(0., 0.02)
    x = layers.Conv2DTranspose(512, 4, 2, padding='same', kernel_initializer=initializer, activation='relu', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Cropping2D(((0, 1), (0, 1)))(x)
    x = layers.Concatenate()([x, concat5])
    initializer = tf.random_normal_initializer(0., 0.02)
    x = layers.Conv2DTranspose(512, 4, 2, padding='same', kernel_initializer=initializer, activation='relu', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Cropping2D(((0, 1), (0, 0)))(x)
    x = layers.Concatenate()([x, concat4])
    initializer = tf.random_normal_initializer(0., 0.02)
    x = layers.Conv2DTranspose(256, 4, 2, padding='same', kernel_initializer=initializer, activation='relu', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Concatenate()([x, concat3])
    initializer = tf.random_normal_initializer(0., 0.02)
    x = layers.Conv2DTranspose(128, 4, 2, padding='same', kernel_initializer=initializer, activation='relu', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Concatenate()([x, concat2])
    initializer = tf.random_normal_initializer(0., 0.02)
    x = layers.Conv2DTranspose(62, 4, 2, padding='same', kernel_initializer=initializer, activation='relu', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Concatenate()([x, concat1])
    initializer = tf.random_normal_initializer(0., 0.02)
    outputs = layers.Conv2DTranspose(1, 4, 2, padding='same', kernel_initializer=initializer, activation='relu')(x)

    return keras.Model(inputs=inputs, outputs=outputs,
                            name="generator")
    
# the discriminator is a patch-classifier
def get_discriminator():
    inp = tf.keras.layers.Input(shape=[720, 1440, 5], name='input_image')
    tar = tf.keras.layers.Input(shape=[720, 1440, 1], name='target_image')
    x = tf.keras.layers.concatenate([inp, tar])
    initializer = tf.random_normal_initializer(0., 0.02)
    down1 = layers.Conv2D(64, 4, 2, activation='relu', padding='same', kernel_initializer=initializer, use_bias=False)(x)
    initializer = tf.random_normal_initializer(0., 0.02)
    down2 = layers.Conv2D(128, 4, 2, activation='relu', padding='same', kernel_initializer=initializer, use_bias=False)(down1)
    down2 = layers.BatchNormalization()(down2)
    initializer = tf.random_normal_initializer(0., 0.02)
    down3 = layers.Conv2D(256, 4, 2, activation='relu', padding='same', kernel_initializer=initializer, use_bias=False)(down2)
    down3 = layers.BatchNormalization()(down3)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1, kernel_initializer=initializer, use_bias=False)(down3)
    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)
    last = tf.keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)(leaky_relu)
    return keras.Model(inputs=[inp, tar], outputs=last, name='discriminator')

# initiate and compile the model
discriminator = get_discriminator()
generator = get_generator()
gan = GAN(discriminator=discriminator, generator=generator)

gan.compile(d_optimizer=tf.keras.optimizers.Adam(2e-4, beta_1=0.5),
        g_optimizer=tf.keras.optimizers.Adam(2e-4, beta_1=0.5),
        d_loss_fn=discriminator_loss,
        g_loss_fn=generator_loss)
