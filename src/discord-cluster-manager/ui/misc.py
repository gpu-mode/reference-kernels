import discord
from discord import Interaction, SelectOption, ui
from utils import send_discord_message


class GPUSelectionView(ui.View):
    def __init__(self, available_gpus: list[str]):
        super().__init__()

        # Add the Select Menu with the list of GPU options
        select = ui.Select(
            placeholder="Select GPUs for this leaderboard...",
            options=[SelectOption(label=gpu, value=gpu) for gpu in available_gpus],
            min_values=1,  # Minimum number of selections
            max_values=len(available_gpus),  # Maximum number of selections
        )
        select.callback = self.select_callback
        self.add_item(select)

    async def select_callback(self, interaction: Interaction):
        # Retrieve the selected options
        select = interaction.data["values"]
        self.selected_gpus = select
        # Acknowledge the interaction
        await interaction.response.defer(ephemeral=True)
        self.stop()


class DeleteConfirmationModal(ui.Modal, title="Confirm Deletion"):
    def __init__(self, field_name: str, field_value: str, db):
        super().__init__()
        self.field_name = field_name
        self.field_value = field_value
        self.db = db
        self.confirmation = ui.TextInput(
            label=f"Type '{field_value}' to confirm deletion",
            placeholder=f"Enter the {field_name}",
            required=True,
        )
        self.add_item(self.confirmation)

    async def on_submit(self, interaction: discord.Interaction):
        if self.confirmation.value == self.field_value:
            with self.db as db:
                method = getattr(db, f"delete_{self.field_name}", None)
                assert method is not None, f"Delete method for {self.field_name} not found in db"
                err = method(self.field_value)
                if err:
                    await send_discord_message(
                        interaction,
                        "An error occurred while deleting the leaderboard.",
                        ephemeral=True,
                    )
                else:
                    await send_discord_message(
                        interaction,
                        f"{self.field_name} '{self.field_value}' deleted.",
                        ephemeral=True,
                    )
        else:
            await send_discord_message(
                interaction,
                f"Deletion cancelled: The {self.field_name} didn't match.",
                ephemeral=True,
            )


def create_delete_confirmation_modal(field_name: str, field_value: str, db):
    return DeleteConfirmationModal(field_name, field_value, db)
