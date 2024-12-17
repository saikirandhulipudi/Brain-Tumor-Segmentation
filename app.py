import os
import tensorflow as tf # version 2.4.1
import nibabel as nib
import matplotlib.pyplot as plt
from flask import Flask, render_template, request

slice_num = 60
cur_dirr = os.path.abspath(os.getcwd())

START_AT = 22
NUM_SLICES = 100
IMG_SIZE = 128

CLASSES = {
    0: 'NOT tumor',
    1: 'NECROTIC/CORE',  # or NON-ENHANCING tumor CORE
    2: 'EDEMA',
    3: 'ENHANCING'  # original 4 -> converted into 3 later
}


def preprocess(flair_path, ce_path):
    flair = nib.load(flair_path).get_fdata()[..., START_AT:START_AT+NUM_SLICES]
    t1ce = nib.load(ce_path).get_fdata()[..., START_AT:START_AT+NUM_SLICES]
    flair = tf.convert_to_tensor(flair)
    t1ce = tf.convert_to_tensor(t1ce)
    flair = tf.transpose(flair, [2, 0, 1])
    t1ce = tf.transpose(t1ce, [2, 0, 1])

    X = tf.concat([
        tf.expand_dims(flair, -1),
        tf.expand_dims(t1ce, -1)
    ], axis=-1)

    X = tf.image.resize(X, (IMG_SIZE, IMG_SIZE))
    return X


def predict_tumors(X, dropdown):

    pred = unet.predict(tf.expand_dims(X, 0)/tf.reduce_max(X)), 
    pred = tf.squeeze(pred, 0)

    plt.imshow(X[..., 0], cmap='gray', alpha=0.3)

    if dropdown == CLASSES[1]:
        plt.imshow(pred[..., 1], cmap="OrRd", alpha=0.7)
        plt.title(f'{CLASSES[1]} predicted')
        plt.axis(False)

    elif dropdown == CLASSES[2]:
        plt.imshow(pred[..., 2], cmap="OrRd", alpha=0.7)
        plt.title(f'{CLASSES[2]} predicted')
        plt.axis(False)

    elif dropdown == CLASSES[3]:
        plt.imshow(pred[..., 3], cmap="OrRd", alpha=0.7)
        plt.title(f'{CLASSES[3]} predicted')
        plt.axis(False)

    else:
        plt.imshow(tf.argmax(pred, axis=-1), cmap="Reds", alpha=0.7)
        plt.title("Predicted")
        plt.axis(False)

    plt.tight_layout()
    plt.savefig(cur_dirr + '\\static\\out_img.png')
    # plt.show()


menuu = list(CLASSES.values())[1:]

unet = tf.keras.models.load_model("unet_brats.h5", compile=False)
print('\nready to launch...\n')


app = Flask(__name__)


@app.route('/')
def home():
    return render_template("home.html", menuu=menuu)


@app.route('/outpg', methods=["GET", "POST"])
def outtputs():

    flair_path = os.path.abspath(request.form.get("flair_path"))
    ce_path = os.path.abspath(request.form.get("ce_path"))
    select_tumor = (request.form.get("part_of_tumor"))

    if (not flair_path):
        return render_template("failure.html", message="missing flair path")
    if (not ce_path):
        return render_template("failure.html", message="missing ce path")
    if (select_tumor not in menuu and select_tumor != 'all_the_tumor'):
        return render_template("failure.html", message="invalid tumor part")

    x = preprocess(flair_path, ce_path)[slice_num]

    plt.imshow(x[..., 0], cmap='gray')
    plt.title('flair input image')
    plt.axis(False)
    plt.tight_layout()
    plt.savefig(cur_dirr + '\\static\\input1.png')

    plt.imshow(x[..., 1], cmap='gray')
    plt.title('ce input image')
    plt.axis(False)
    plt.tight_layout()
    plt.savefig(cur_dirr + '\\static\\input2.png')

    predict_tumors(X=x, dropdown=select_tumor)

    return render_template("success.html",
                           flair_path="input1.png",
                           ce_path="input2.png",
                           output_path="out_img.png")


if __name__ == '__main__':
    app.run(debug=True)
