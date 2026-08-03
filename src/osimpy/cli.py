from pydantic import BaseModel, Field
from pydantic_settings import CliApp, CliSubCommand

from osimpy.tools.scale import ScaleSettings
from osimpy.tools.ik import IKSettings
from osimpy.tools.id import IDSettings
from osimpy.tools.cmc import CMCSettings
from osimpy.moco.inverse import MocoInverseSettings
from osimpy.moco.track import MocoTrackSettings


class OsimPyCli(BaseModel):
    scale: CliSubCommand[ScaleSettings] = Field(alias="scale")
    ik: CliSubCommand[IKSettings] = Field(alias="ik")
    id: CliSubCommand[IDSettings] = Field(alias="id")
    cmc: CliSubCommand[CMCSettings] = Field(alias="cmc")
    moco_inverse: CliSubCommand[MocoInverseSettings] = Field(alias="moco-inverse")
    moco_track: CliSubCommand[MocoTrackSettings] = Field(alias="moco-track")

    def cli_cmd(self) -> None:
        CliApp.run_subcommand(self)


def main() -> None:
    CliApp.run(OsimPyCli)
