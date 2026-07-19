class RootPath:
    def __init__(self, waypoints):
        self.waypoints = waypoints

    def __repr__(self):
        return f"RootPath(waypoints={self.waypoints})"