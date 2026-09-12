using System;
using UnityEngine;

namespace AVRS {
    [Serializable]
    public struct HistoricalSceneState {
        public double julianDay;
        public double latitude;
        public double longitude;
        public float timeSeconds;
    }

    public sealed class HistoricalSceneBridge : MonoBehaviour {
        public HistoricalSceneState State;
        public void SetHistoricalTime(double julianDay) => State.julianDay = julianDay;
        public void SetLocation(double latitude, double longitude) {
            State.latitude = latitude; State.longitude = longitude;
        }
    }
}
