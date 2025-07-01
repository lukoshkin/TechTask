from loguru import logger

from src.eval.datagen.synthetic import SyntheticDataset
from src.models import TestConfig


def main() -> None:
    test_cfg_file = "test-config.yaml"
    test_cfg = TestConfig.from_yaml(test_cfg_file)
    logger.debug(f"Loaded configuration: {test_cfg}")

    dgen = SyntheticDataset(test_cfg.datagen)
    dgen.sort_by_lang(test_cfg.chunked_data)
    dset = dgen.generate_dataset()
    print(dset)


if __name__ == "__main__":
    main()
