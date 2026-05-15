from pathlib import Path

from kaggle_otto2.config.enums import EnvEnum


class DirConfig:
    def __init__(self, exp: str, env: EnvEnum):
        self.root_dir = self._get_root_dir(env)
        self.data_root_dir = self._get_data_root_dir(env)
        self.input_dir = self._get_input_dir(env)
        self.output_dir = self._get_output_dir(env)
        self.exp_output_dir = self.output_dir / exp
        self._mkdirs(env)

    def _mkdirs(self, env: str):
        self.exp_output_dir.mkdir(parents=True, exist_ok=True)

    def _get_root_dir(self, env: str):
        if env == "colab":
            return Path("/content/kaggle-otto")
        return Path(__file__).parents[2]

    def _get_data_root_dir(self, env: str):
        if env == "colab":
            return Path("/content/drive/MyDrive/kaggle/kaggle-otto")
        if env == "kaggle":
            return Path("/kaggle")
        root_dir = self._get_root_dir(env)
        return root_dir

    def _get_input_dir(self, env: str):
        if env == "kaggle":
            return Path("/kaggle/input")
        return self.data_root_dir / "input"

    def _get_output_dir(self, env: str):
        if env == "kaggle":
            return Path("/kaggle/working/output")
        return self.data_root_dir / "output"


if __name__ == "__main__":
    dir_config = DirConfig("exp001", "local")

    print(dir_config.root_dir)
    print(dir_config.data_root_dir)
    print(dir_config.input_dir)
    print(dir_config.output_dir)
    print(dir_config.exp_output_dir)

    assert str(dir_config.root_dir).endswith("kaggle-otto2")
    assert dir_config.data_root_dir == dir_config.root_dir
    assert dir_config.input_dir == dir_config.root_dir / "input"
    assert dir_config.output_dir == dir_config.root_dir / "output"
    assert dir_config.exp_output_dir == dir_config.root_dir / "output" / "exp001"
