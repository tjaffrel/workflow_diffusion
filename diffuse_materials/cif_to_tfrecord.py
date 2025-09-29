#!/usr/bin/env python3
"""
Convert CIF files to TFRecord format for MOF diffusion training.
"""

import argparse
import os
import glob
import tensorflow as tf
from pymatgen.core import Structure


def parse_cif_file(cif_path):
    """Parse CIF file and extract required data."""
    structure = Structure.from_file(cif_path)
    
    # Extract data
    frac_coords = [site.frac_coords for site in structure]
    atom_types = [site.species.elements[0].Z for site in structure]
    lengths = [structure.lattice.a, structure.lattice.b, structure.lattice.c]
    angles = [structure.lattice.alpha, structure.lattice.beta, structure.lattice.gamma]
    formula = structure.composition.reduced_formula
    
    # Convert to TensorFlow tensors
    import numpy as np
    frac_coords = tf.constant(np.array(frac_coords), dtype=tf.float32)
    atom_types = tf.constant(np.array(atom_types), dtype=tf.int64)
    lengths = tf.constant(np.array(lengths), dtype=tf.float32)
    angles = tf.constant(np.array(angles), dtype=tf.float32)
    
    return frac_coords, atom_types, lengths, angles, formula


def create_tfrecord_example(frac_coords, atom_types, lengths, angles, formula):
    """Create TFRecord example."""
    # Flatten tensors and convert to lists
    frac_coords_flat = tf.reshape(frac_coords, [-1]).numpy().tolist()
    atom_types_flat = atom_types.numpy().tolist()
    lengths_flat = lengths.numpy().tolist()
    angles_flat = angles.numpy().tolist()
    
    features = {
        'frac_coords': tf.train.Feature(float_list=tf.train.FloatList(value=frac_coords_flat)),
        'atom_types': tf.train.Feature(int64_list=tf.train.Int64List(value=atom_types_flat)),
        'lengths': tf.train.Feature(float_list=tf.train.FloatList(value=lengths_flat)),
        'angles': tf.train.Feature(float_list=tf.train.FloatList(value=angles_flat)),
        'formula': tf.train.Feature(bytes_list=tf.train.BytesList(value=[formula.encode('utf-8')])),
    }
    
    return tf.train.Example(features=tf.train.Features(feature=features))


def convert_cif_to_tfrecord(cif_dir, output_path):
    """Convert CIF files to TFRecord."""
    cif_files = glob.glob(os.path.join(cif_dir, "*.cif")) + glob.glob(os.path.join(cif_dir, "*.CIF"))
    
    print(f"Processing {len(cif_files)} CIF files")
    
    with tf.io.TFRecordWriter(output_path) as writer:
        for cif_file in cif_files:
            frac_coords, atom_types, lengths, angles, formula = parse_cif_file(cif_file)
            example = create_tfrecord_example(frac_coords, atom_types, lengths, angles, formula)
            writer.write(example.SerializeToString())
    
    print(f"Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert CIF files to TFRecord")
    parser.add_argument("--cif_dir", required=True, help="Directory with CIF files")
    parser.add_argument("--output", required=True, help="Output TFRecord file")
    
    args = parser.parse_args()
    convert_cif_to_tfrecord(args.cif_dir, args.output)


if __name__ == "__main__":
    main()