"""Tests for CLI argument parser (does not require I/O)."""

from __future__ import annotations

import pytest

from packetcore.cli import build_parser


def test_parser_train_required_args():
    parser = build_parser()
    args = parser.parse_args(["train", "--data", "x.csv", "--artifact-dir", "out/"])
    assert args.command == "train"
    assert args.data == "x.csv"
    assert args.artifact_dir == "out/"


def test_parser_train_defaults():
    parser = build_parser()
    args = parser.parse_args(["train", "--data", "x.csv", "--artifact-dir", "out/"])
    assert args.val_frac == 0.15
    assert args.test_frac == 0.15
    assert args.epochs == 50
    assert args.batch_size == 256
    assert args.latent_dim == 16
    assert args.window_size == 32
    assert args.device == "cpu"


def test_parser_calibrate():
    parser = build_parser()
    args = parser.parse_args(["calibrate", "--artifact-dir", "out/"])
    assert args.command == "calibrate"
    assert args.target_fpr == 0.01


def test_parser_infer():
    parser = build_parser()
    args = parser.parse_args(["infer", "--data", "new.csv", "--artifact-dir", "out/", "--output", "res.csv"])
    assert args.command == "infer"
    assert args.output == "res.csv"


def test_parser_missing_subcommand():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_parser_train_missing_data():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["train", "--artifact-dir", "out/"])
