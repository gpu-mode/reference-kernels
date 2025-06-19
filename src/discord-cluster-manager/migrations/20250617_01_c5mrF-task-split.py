"""
Split off non-runner related things into separate db columns.
"""
__depends__ = {'20250506_01_38PkG-add-index-on-runs-runner-score'}

import json
from yoyo import step


def apply_step(conn):
    cursor = conn.cursor()

    # Add new table for code templates
    cursor.execute("""
            CREATE TABLE IF NOT EXISTS leaderboard.templates (
                id SERIAL PRIMARY KEY,
                leaderboard_id INTEGER NOT NULL REFERENCES leaderboard.leaderboard(id),
                lang TEXT NOT NULL,
                code TEXT NOT NULL
            )
            """)

    # Extract the description column
    cursor.execute("""
        ALTER TABLE leaderboard.leaderboard 
        ADD COLUMN description TEXT,
    """)

    # Extract data from JSON column and populate new columns
    cursor.execute("SELECT id, task FROM leaderboard.leaderboard")
    rows = cursor.fetchall()

    for row in rows:
        row_id, json_data = row
        if json_data:
            try:
                parsed_data = json.loads(json_data) if isinstance(json_data, str) else json_data
                description = parsed_data.get('description', '')
                templates = parsed_data.get('templates', {})

                # Remove description and templates from original JSON
                parsed_data.pop('description', None)
                parsed_data.pop('templates', None)

                cursor.execute("""
                    UPDATE leaderboard.leaderboard 
                    SET description = %s, task = %s
                    WHERE id = %s
                """, (description, json.dumps(templates), json.dumps(parsed_data), row_id))

                for lang, code in templates.items():
                    cursor.execute(
                        """
                        INSERT INTO leaderboard.templates (leaderboard_id, lang, code)
                        VALUES (%s, %s, %s)
                        """,
                        (row_id, lang, code),
                    )
            except (json.JSONDecodeError, TypeError) as e:
                print(f"Failed to parse JSON for row {row_id}: {e}")

    # Now make description NOT NULL
    cursor.execute("ALTER TABLE leaderboard.leaderboard ALTER COLUMN description SET NOT NULL")

    conn.commit()


def rollback_step(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT id, description, task FROM leaderboard.leaderboard")
    for lb_id, description, task_json in cursor.fetchall():
        task_data = json.loads(task_json)
        task_data['description'] = description

        cursor.execute("""
            SELECT lang, code 
            FROM leaderboard.templates 
            WHERE leaderboard_id = %s
        """, (lb_id,))

        templates = {}
        for lang, code in cursor.fetchall():
            templates[lang] = code
        task_data['templates'] = templates

        cursor.execute("""
            UPDATE leaderboard.leaderboard 
            SET task = %s
            WHERE id = %s
        """, (json.dumps(task_data), lb_id))

    cursor.execute("ALTER TABLE leaderboard.leaderboard DROP COLUMN description")
    cursor.execute("DROP TABLE IF EXISTS leaderboard.templates")

    conn.commit()


steps = [
    step(apply_step, rollback_step)
]
