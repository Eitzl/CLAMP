"""Runtime settings, overridable via .env or CLAMP_* environment variables.

See MD_design_docs/09_phase1_implementation_design.md §4.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CLAMP_", env_file=".env")

    data_root: Path = Path("data")

    dbaasp_base_url: str = "https://dbaasp.org"
    dbaasp_requests_per_second: float = 3.0
    dbaasp_page_size: int = 200

    http_timeout_s: float = 30.0

    @property
    def raw_dir(self) -> Path:
        return self.data_root / "raw"

    @property
    def interim_dir(self) -> Path:
        return self.data_root / "interim"

    @property
    def processed_dir(self) -> Path:
        return self.data_root / "processed"

    @property
    def datasheet_dir(self) -> Path:
        return self.data_root / "datasheet"

    @property
    def manifest_path(self) -> Path:
        return self.data_root / "manifest.json"


settings = Settings()
