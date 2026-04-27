import numpy as np
import torch.utils.data as data


def produce_adjacent_matrix_2_neighbors(flag_bits, stroke_len, pen_down_id):
    seq_len = flag_bits.shape[0]
    assert flag_bits.shape == (seq_len, 1)
    adja_matr = np.full((seq_len, seq_len), -1e10, dtype=np.float32)

    adja_matr[0][0] = 0
    if stroke_len >= 2 and flag_bits[0] == pen_down_id:
        adja_matr[0][1] = 0

    for idx in range(1, stroke_len):
        adja_matr[idx][idx] = 0

        if flag_bits[idx - 1] == pen_down_id:
            adja_matr[idx][idx - 1] = 0

        if idx == stroke_len - 1:
            break

        if flag_bits[idx] == pen_down_id:
            adja_matr[idx][idx + 1] = 0

    return adja_matr


def produce_adjacent_matrix_4_neighbors(flag_bits, stroke_len, pen_down_id):
    seq_len = flag_bits.shape[0]
    assert flag_bits.shape == (seq_len, 1)
    adja_matr = np.full((seq_len, seq_len), -1e10, dtype=np.float32)

    adja_matr[0][0] = 0
    if stroke_len >= 2 and flag_bits[0] == pen_down_id:
        adja_matr[0][1] = 0
        if stroke_len >= 3 and flag_bits[1] == pen_down_id:
            adja_matr[0][2] = 0

    for idx in range(1, stroke_len):
        adja_matr[idx][idx] = 0

        if flag_bits[idx - 1] == pen_down_id:
            adja_matr[idx][idx - 1] = 0
            if idx >= 2 and flag_bits[idx - 2] == pen_down_id:
                adja_matr[idx][idx - 2] = 0

        if idx == stroke_len - 1:
            break

        if idx <= stroke_len - 2 and flag_bits[idx] == pen_down_id:
            adja_matr[idx][idx + 1] = 0
            if idx <= stroke_len - 3 and flag_bits[idx + 1] == pen_down_id:
                adja_matr[idx][idx + 2] = 0

    return adja_matr


def produce_adjacent_matrix_joint_neighbors(
    flag_bits,
    stroke_len,
    pen_down_id,
    pen_up_id,
):
    seq_len = flag_bits.shape[0]
    assert flag_bits.shape == (seq_len, 1)
    adja_matr = np.full((seq_len, seq_len), -1e10, dtype=np.float32)

    adja_matr[0][0] = 0
    adja_matr[0][stroke_len - 1] = 0
    adja_matr[stroke_len - 1][stroke_len - 1] = 0
    adja_matr[stroke_len - 1][0] = 0

    assert flag_bits[0] == pen_down_id or flag_bits[0] == pen_up_id

    if flag_bits[0] == pen_up_id and stroke_len >= 2:
        adja_matr[0][1] = 0

    for idx in range(1, stroke_len):
        assert flag_bits[idx] == pen_down_id or flag_bits[idx] == pen_up_id

        adja_matr[idx][idx] = 0

        if flag_bits[idx - 1] == pen_up_id:
            adja_matr[idx][idx - 1] = 0

        if idx == stroke_len - 1:
            break

        if idx <= stroke_len - 2 and flag_bits[idx] == pen_up_id:
            adja_matr[idx][idx + 1] = 0

    return adja_matr


def generate_padding_mask(stroke_length, max_seq_len):
    padding_mask = np.ones((max_seq_len, 1), dtype=np.int64)
    padding_mask[stroke_length:, :] = 0
    return padding_mask


class JointDataset_2nn4nnjnn(data.Dataset):
    def __init__(
        self,
        sketch_list,
        data_dict,
        max_seq_len,
        pen_down_id,
        pen_up_id,
    ):
        with open(sketch_list) as sketch_url_file:
            sketch_url_list = sketch_url_file.readlines()

        self.coordinate_urls = [
            sketch_url.strip().split(" ")[0] for sketch_url in sketch_url_list
        ]
        self.labels = [
            int(sketch_url.strip().split(" ")[-1]) for sketch_url in sketch_url_list
        ]
        self.data_dict = data_dict
        self.max_seq_len = int(max_seq_len)
        self.pen_down_id = int(pen_down_id)
        self.pen_up_id = int(pen_up_id)

    def __len__(self):
        return len(self.coordinate_urls)

    def __getitem__(self, item):
        coordinate_url = self.coordinate_urls[item]
        label = self.labels[item]
        coordinate, flag_bits, stroke_len = self.data_dict[coordinate_url]

        attention_mask_2_neighbors = produce_adjacent_matrix_2_neighbors(
            flag_bits, stroke_len, self.pen_down_id
        )
        attention_mask_4_neighbors = produce_adjacent_matrix_4_neighbors(
            flag_bits, stroke_len, self.pen_down_id
        )
        attention_mask_joint_neighbors = produce_adjacent_matrix_joint_neighbors(
            flag_bits, stroke_len, self.pen_down_id, self.pen_up_id
        )
        padding_mask = generate_padding_mask(stroke_len, self.max_seq_len)

        position_encoding = np.arange(self.max_seq_len, dtype=np.int64)
        position_encoding.resize([self.max_seq_len, 1])

        assert coordinate.shape == (self.max_seq_len, 2)
        assert flag_bits.shape == (self.max_seq_len, 1)

        coordinate = coordinate.astype("float32")
        return (
            coordinate,
            label,
            flag_bits.astype("int"),
            stroke_len,
            attention_mask_2_neighbors,
            attention_mask_4_neighbors,
            attention_mask_joint_neighbors,
            padding_mask,
            position_encoding,
        )
