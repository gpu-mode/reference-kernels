"""
Fix code table
Uses bytea to store user-submitted code so we're safe to have any sort of special characters.
"""

from yoyo import step

__depends__ = {"20250617_01_c5mrF-task-split"}

"""
Yoyo migration to convert code_files table from TEXT to BYTEA
"""


def convert_code_to_bytea(conn):
    """Convert existing TEXT code to BYTEA and recalculate hashes"""
    cursor = conn.cursor()

    # Get all existing records
    cursor.execute("SELECT id, old_code FROM leaderboard.code_files")
    records = cursor.fetchall()

    existing_codes = {}
    num_duplicates = 0
    num_encoded = 0

    for record_id, code_text in records:
        # broken with the old code
        if code_text.startswith("\\x"):
            num_encoded += 1
            code_text = bytes.fromhex(code_text[2:]).decode("utf-8")
        code_bytes = code_text.encode("utf-8")
        # with the old broken code and experimentation, it is possible that we got some
        # duplicates; fix this here
        if code_bytes in existing_codes:
            cursor.execute(
                "UPDATE leaderboard.submission SET code_id = %s WHERE code_id = %s",
                (existing_codes[code_bytes], record_id),
            )
            cursor.execute("DELETE FROM leaderboard.code_files WHERE id = %s", (record_id,))
            num_duplicates += 1
            continue

        existing_codes[code_bytes] = record_id

        # Update record with bytea and new hash
        cursor.execute(
            "UPDATE leaderboard.code_files SET code = %s WHERE id = %s", (code_bytes, record_id)
        )

    print(f"Found and removed {num_duplicates} duplicates")
    print(f"Reencoded {num_encoded} code submissions")


def convert_bytea_to_text(conn):
    """Convert existing BYTEA code to TEXT and recalculate hashes"""
    cursor = conn.cursor()
    # Get all existing records
    cursor.execute("SELECT id, code FROM leaderboard.code_files")
    records = cursor.fetchall()

    for record_id, code_bytes in records:
        code_text = bytes(code_bytes).decode("utf-8")
        cursor.execute(
            "UPDATE leaderboard.code_files SET old_code = %s WHERE id = %s",
            (code_text.encode("utf-8"), record_id),
        )


steps = [
    # prepare the table columns
    step(
        """
        ALTER TABLE leaderboard.code_files DROP COLUMN hash;
        ALTER TABLE leaderboard.code_files RENAME COLUMN code TO old_code;
        ALTER TABLE leaderboard.code_files ADD COLUMN code BYTEA NOT NULL DEFAULT '';
        """,
        """
         ALTER TABLE leaderboard.code_files DROP COLUMN code;
         ALTER TABLE leaderboard.code_files RENAME COLUMN old_code TO code;
         ALTER TABLE leaderboard.code_files ADD COLUMN hash TEXT
            GENERATED ALWAYS AS (encode(sha256(code::bytea), 'hex')) STORED;
         """,
    ),
    # run the conversion
    step(convert_code_to_bytea, convert_bytea_to_text),
    # clean up the table and reintroduce hashes
    # ALTER TABLE leaderboard.code_files DROP COLUMN old_code;
    # do this later, once we're confident that the migration works
    step(
        """
       ALTER TABLE leaderboard.code_files ALTER COLUMN old_code DROP NOT NULL;
       ALTER TABLE leaderboard.code_files ADD COLUMN hash TEXT
           GENERATED ALWAYS AS (encode(sha256(code), 'hex')) STORED NOT NULL UNIQUE;
       ALTER TABLE leaderboard.code_files ALTER COLUMN code DROP DEFAULT;
       """,
        """
         ALTER TABLE leaderboard.code_files DROP COLUMN hash;
         """,
    ),
]
