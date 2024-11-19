// To run
// npm install discord.js
// node nuke_commands.js

const { REST, Routes } = require('discord.js');

// Bot IDs, Guild ID, and tokens
require('dotenv').config();

const clientId_staging_bot = process.env.DISCORD_DEBUG_TOKEN.split('.')[0];
const clientId_prod_bot = process.env.DISCORD_TOKEN.split('.')[0];
const guildId = process.env.DISCORD_MARK_STAGING_ID;

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
