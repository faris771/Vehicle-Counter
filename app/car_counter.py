class CarCounter:
    def __init__(self, limit_line: dict):
        self.limit_line = limit_line
        self.counter = 0
        self.crossed_ids = {}
        self.previous_centers = {}

    def check_and_count(self, id, cx, cy):
        prev = self.previous_centers.get(id)
        self.previous_centers[id] = (cx, cy)

        if prev:
            _, prev_cy = prev
            line_y = self.limit_line['y1']
            if prev_cy < line_y and cy >= line_y:
                if id not in self.crossed_ids:
                    self.crossed_ids[id] = True
                    self.counter += 1

    def get_count(self):
        return self.counter
