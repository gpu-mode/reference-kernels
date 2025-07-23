import textwrap
from typing import Any, Dict, List, Optional

import discord

DISCORD_MAX_EMBED_WIDTH = 56


class TableView(discord.ui.View):
    def __init__(
        self,
        data: List[Dict[str, Any]],
        items_per_page: int = 10,
        column_widths: Dict[str, int] = None,
        padding_width: int = 3,
    ):
        super().__init__()
        self.data = data
        self.current_page = 0
        self.items_per_page = items_per_page
        self.total_pages = max(1, (len(data) + items_per_page - 1) // items_per_page)
        self.column_widths = column_widths
        self.padding_width = padding_width
        self.update_buttons()

    def update_buttons(self):
        self.previous_page.disabled = self.current_page == 0
        self.next_page.disabled = self.current_page >= self.total_pages - 1
        self.page_counter.label = f"Page {self.current_page + 1}/{self.total_pages}"

    @discord.ui.button(label="◀", style=discord.ButtonStyle.primary)
    async def previous_page(self, interaction: discord.Interaction, button: discord.ui.Button):
        self.current_page = max(0, self.current_page - 1)
        self.update_buttons()
        await interaction.response.edit_message(
            embed=create_table_page(
                self.data,
                self.current_page,
                self.items_per_page,
                self.column_widths,
                self.padding_width,
            ),
            view=self,
        )

    @discord.ui.button(label="Page 1/1", style=discord.ButtonStyle.secondary, disabled=True)
    async def page_counter(self, interaction: discord.Interaction, button: discord.ui.Button):
        pass

    @discord.ui.button(label="▶", style=discord.ButtonStyle.primary)
    async def next_page(self, interaction: discord.Interaction, button: discord.ui.Button):
        self.current_page = min(self.total_pages - 1, self.current_page + 1)
        self.update_buttons()

        await interaction.response.edit_message(
            embed=create_table_page(
                self.data,
                self.current_page,
                self.items_per_page,
                self.column_widths,
                self.padding_width,
            ),
            view=self,
        )


def create_table_page(
    data: List[Dict[str, Any]],
    page: int,
    items_per_page: int,
    column_widths: Optional[Dict[str, int]],
    padding_width: int,
) -> discord.Embed:
    if not data:
        return discord.Embed(description="No data to display")

    padding = " " * padding_width

    if column_widths is None:
        remaining_width = DISCORD_MAX_EMBED_WIDTH - len(padding) * len(data[0].keys())
        column_widths = {
            column: remaining_width // len(data[0].keys()) for column in data[0].keys()
        }

    if sum(column_widths.values()) + len(padding) * len(data[0].keys()) > DISCORD_MAX_EMBED_WIDTH:
        raise ValueError(
            """Column widths exceed the maximum embed width.
            Please provide smaller padding_width or column_widths"""
        )

    start_idx = page * items_per_page
    end_idx = min(start_idx + items_per_page, len(data))

    page_data = data[start_idx:end_idx]
    column_names = list(data[0].keys())

    headers = [
        f"{column_name:<{column_widths[column_name]}}{padding}" for column_name in column_names
    ]

    header = "".join(headers)
    divider = "-" * (sum(column_widths.values()) + len(padding) * (len(headers)))

    table_rows = [header, divider]

    for item in page_data:
        wrapped_columns = {
            column: textwrap.wrap(str(item[column]), column_widths[column])
            for column in column_names
        }

        max_lines = max(len(lines) for lines in wrapped_columns.values())
        max_lines = max(max_lines, 1)

        for i in range(max_lines):
            row_parts = []
            for column_name in column_names:
                lines = wrapped_columns[column_name]
                part = lines[i] if i < len(lines) else ""
                row_parts.append(f"{part:<{column_widths[column_name]}}{padding}")

            table_rows.append("".join(row_parts))

    return discord.Embed(description=f"```\n{'\n'.join(table_rows)}\n```")


def create_table(
    title: str,
    data: List[Dict[str, Any]],
    items_per_page: int = 10,
    column_widths: Dict[str, int] = None,
    padding_width: int = 3,
) -> tuple[discord.Embed, TableView]:
    """
    Create a paginated table for Discord with navigation buttons.

    Args:
        title (str): The title of the table
        data (List[Dict[str, Any]]): List of dictionaries where each dictionary represents a row
        items_per_page (int, optional): Number of items to display per page. Defaults to 10.

    Returns:
        tuple[discord.Embed, TableView]: The embed containing the table and the view with navigation
        buttons
    """
    if not data:
        return discord.Embed(title=title, description="No data to display"), None
    view = TableView(data, items_per_page, column_widths, padding_width)
    embed = create_table_page(data, 0, items_per_page, column_widths, padding_width)
    embed.title = title

    return embed, view
