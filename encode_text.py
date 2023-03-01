import tensorflow as tf
import clip
from datasets import load_dataset
from tqdm.auto import tqdm
import pyarrow
import pyarrow.feather as feather
import tensorflow_text
import tensorflow_hub as hub
import numpy as np

physical_devices = tf.config.list_physical_devices("GPU")
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass


BATCH_SIZE = 128
SHARD_SIZE = 5_000


device = "cuda"

# teacher
teacher, preprocess = clip.load("ViT-B/32", device=device)

# student
text_input = tf.keras.layers.Input(shape=(), dtype=tf.string)
preprocessor = hub.KerasLayer(
    "https://tfhub.dev/tensorflow/bert_multi_cased_preprocess/3"
)
encoder_inputs = preprocessor(text_input)
encoder = hub.KerasLayer(
    "https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/4", trainable=False
)
outputs = encoder(encoder_inputs)
pooled_output = outputs["pooled_output"]
text_encoder = tf.keras.Model(text_input, pooled_output)


output_contents = {
    "teacher_embeddings": [],
    "student_embeddings": [],
}


def reencode(batch):
    batch["caption"] = batch["caption"].encode("ascii", "replace")
    return {"caption": batch["caption"]}


dataset = load_dataset("conceptual_captions")["train"]
dataset = dataset.map(reencode, num_proc=16)
dataset = dataset.to_tf_dataset(columns=["caption"], batch_size=BATCH_SIZE)
dataset = dataset.take(SHARD_SIZE)
dataset = dataset.prefetch(tf.data.AUTOTUNE)


def wrapper(gen):
    while True:
        try:
            yield next(gen)
        except StopIteration:
            break
        except Exception as e:
            pass


for i, caption in enumerate(
    tqdm(wrapper(dataset.as_numpy_iterator()), total=SHARD_SIZE)
):
    try:
        caption = caption.astype("U13")
    except Exception as e:
        cast_fails += 1
    teacher_embedding = (
        teacher.encode_text(clip.tokenize(caption).to(device))
        .to("cpu")
        .detach()
        .numpy()
    )
    student_embedding = text_encoder(np.array(caption))
    output_contents["teacher_embeddings"].append(teacher_embedding)
    output_contents["student_embeddings"].append(student_embedding)

print(f"encountered a total of {cast_fails} cast fails.")

table = pyarrow.Table.from_pydict(output_contents)
feather.write_feather(
    table,
    "encoded_text_01.feather",
    compression="uncompressed",
)
