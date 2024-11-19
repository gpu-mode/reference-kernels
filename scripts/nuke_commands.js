// To run
// npm install discord.js dotenv
// node nuke_commands.js

const { REST, Routes } = require('discord.js');
require('dotenv').config();

// Bot IDs, Guild ID, and tokens from environment variables
const clientId_staging_bot = process.env.DISCORD_STAGING_CLIENT_ID;
const clientId_prod_bot = process.env.DISCORD_PROD_CLIENT_ID;
const guildId = process.env.GUILD_ID;

const stagingToken = process.env.DISCORD_DEBUG_TOKEN;
const prodToken = process.env.DISCORD_TOKEN;

async function deleteCommands(clientId, token) {
    const rest = new REST().setToken(token);
    
    try {
        // Delete guild commands
        await rest.put(
            Routes.applicationGuildCommands(clientId, guildId),
            { body: [] }
        );
        console.log(`Successfully deleted all guild commands for bot ${clientId}`);

        // Delete global commands
        await rest.put(
            Routes.applicationCommands(clientId),
            { body: [] }
        );
        console.log(`Successfully deleted all global commands for bot ${clientId}`);
    } catch (error) {
        console.error(`Error for bot ${clientId}:`, error);
    }
}

// Delete commands for both bots (or just staging if no prod token)
async function deleteAllCommands() {
    await deleteCommands(clientId_staging_bot, stagingToken);
    if (prodToken) {
        await deleteCommands(clientId_prod_bot, prodToken);
    }
}

// Run the script
deleteAllCommands();
