import tempfile
import unittest
from pathlib import Path

from ancient_objects_db import ObjectDatabase
from thamudic_web_app import create_app


class ThamudicWebAppTests(unittest.TestCase):
    def test_dashboard_and_catalog_use_database(self):
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "objects.sqlite"
            db = ObjectDatabase(db_path)
            db.add_object({
                "title": "Test inscription",
                "period_key": "iron_age",
                "object_type": "inscription",
                "script_key": "old_north_arabian",
                "source_name": "Test source",
                "tags": ["thamudic"],
            })
            db.close()

            app = create_app(str(db_path))
            app.testing = True
            client = app.test_client()

            dashboard = client.get("/")
            self.assertEqual(dashboard.status_code, 200)
            self.assertIn(b"Thamudic Scanner", dashboard.data)

            catalog = client.get("/api/objects?q=Test")
            self.assertEqual(catalog.status_code, 200)
            self.assertIn(b"Test inscription", catalog.data)

    def test_search_endpoint_filters_by_script(self):
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "objects.sqlite"
            db = ObjectDatabase(db_path)
            db.add_object({"title": "ONA", "script_key": "old_north_arabian"})
            db.add_object({"title": "Other", "script_key": "safaitic"})
            db.close()

            app = create_app(str(db_path))
            app.testing = True
            response = app.test_client().get(
                "/api/objects?script_key=old_north_arabian"
            )
            self.assertEqual(response.status_code, 200)
            payload = response.get_json()
            self.assertEqual(len(payload["objects"]), 1)
            self.assertEqual(payload["objects"][0]["title"], "ONA")


if __name__ == "__main__":
    unittest.main()
