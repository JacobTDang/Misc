# team_assigner.py
import numpy as np
from sklearn.cluster import KMeans

class TeamAssigner:
    def __init__(self):
        self.team_colors      = {}
        self.player_team_dict = {}

    def assign_team_color(self, frame, player_detections):
        colors = []
        for pid, det in player_detections.items():
            colors.append(self.get_player_color(frame, det['bbox']))

        colors = np.vstack(colors)
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
        kmeans.fit(colors)
        self.kmeans = kmeans
        # cluster_centers_ is shape (2,3) → RGB for each team
        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

    def get_player_color(self, frame, bbox):
        # unpack & crop
        x1,y1,x2,y2 = map(int, bbox)
        img = frame[y1:y2, x1:x2]
        top = img[:img.shape[0]//2, :]

        # flatten to (pixels, 3)
        pix2d = top.reshape(-1, 3)
        km = KMeans(n_clusters=2, init="k-means++", n_init=10).fit(pix2d)
        lbls = km.labels_

        # assume background is the majority label in the corners
        h, w = top.shape[:2]
        corners = np.concatenate([
            lbls[0:w],          # top row
            lbls[-w:],          # bottom row
            lbls[::w],          # left col
            lbls[w-1::w]        # right col
        ])
        bg = np.bincount(corners).argmax()
        fg = 1 - bg
        return km.cluster_centers_[fg]

    def get_player_teams(self, frame, bbox, player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]
        color = self.get_player_color(frame, bbox)
        # predict returns array, so grab [0] and shift to 1/2
        tid = int(self.kmeans.predict(color.reshape(1,-1))[0]) + 1
        self.player_team_dict[player_id] = tid
        return tid
