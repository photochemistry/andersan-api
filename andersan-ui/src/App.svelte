<script>
    import { onMount } from 'svelte';
    import 'leaflet/dist/leaflet.css';
    import L from 'leaflet';
    import { fetchData, fetchAddress, fetchPtable } from './retrieve.js';
    import Chart from 'chart.js/auto';

    let map;
    let ox_dict;
    let address = ""; // initialize address to empty string
    let addr_dict;
    let ox_array;
    let p_array; // Renamed to p_array from z for consistency
    let p_max;
    let now;
    let X, Y;
    let ptable;
    let myChart; // Add myChart variable

    function findMatchingRowIndex(array, targetValue1, targetValue2) {
        return array.findIndex(row => row[0] === targetValue1 && row[1] === targetValue2);
    }

    function formatTime(date) {
        const hours = date.getHours().toString().padStart(2, '0');
        const minutes = date.getMinutes().toString().padStart(2, '0');
        return `${hours}:${minutes}`;
    }

    onMount(() => {
        // 地図の初期化
        map = L.map('map').setView([0, 0], 13); // 初期位置は適宜設定

        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        }).addTo(map);

        delete L.Icon.Default.prototype._getIconUrl;
        L.Icon.Default.mergeOptions({
            iconRetinaUrl: '/images/marker-icon-2x.png',
            iconUrl: '/images/marker-icon.png',
            shadowUrl: '/images/marker-shadow.png',
        });
    });

    // 現在地を取得して地図を移動する関数
    const moveToCurrentLocation = () => {
        if (!navigator.geolocation) {
            alert('Geolocation is not supported by your browser');
            return;
        }

        const latitude = 35 + 20 / 60 + 8 / 3600;
        const longitude = 139 + 20 / 60 + 58 / 3600;
        map.setView([latitude, longitude], 16); // 現在地に移動してズーム
        L.marker([latitude, longitude]).addTo(map); // 現在地にマーカーを追加
        fetchAddress(longitude, latitude).then(a => { address = a.address; addr_dict = a; });
        let now_aux = new Date();
        fetchData(now_aux).then(result => { ox_dict = result });
        if (ptable === undefined) {
            fetchPtable().then(result => { ptable = result });
        }
    };

    $: {
        if (myChart) {
            myChart.destroy();
        }

        if (ox_dict !== undefined) {
            if (addr_dict !== undefined) {
                X = addr_dict.X;
                Y = addr_dict.Y;
                let row = findMatchingRowIndex(ox_dict.data.XY, X, Y);
                ox_array = [];
                for (let hr = 1; hr <= 24; hr++) {
                    ox_array.push(Math.round(ox_dict.data[`+${hr}`][row]));
                }
                now = new Date(ox_dict.spec.timestamp[0] * 1000);
                now.setMinutes(0);
                now.setSeconds(0);
                now.setMilliseconds(0);
            }
        }

        if (ptable !== undefined) {
            if (ox_array !== undefined) {
                p_array = []; // Initialize p_array here
                let ticks = [];
                for (let hr = 1; hr <= 24; hr++) {
                    let ox = Math.floor(ox_array[hr - 1] / 5) * 5;
                    let b = `(${ox}, ${hr})`;
                    let a = "120";
                    p_array.push(Math.round(ptable[a][b] * 100));
                    ticks.push(hr);
                }
                p_max = Math.max(...p_array)

                let x = ticks.map((hr) => {
                    const futureTime = new Date(now);
                    futureTime.setHours(now.getHours() + hr);
                    return `${formatTime(futureTime)} (+${hr})`;
                });

                let y1 = ox_array;
                let y2 = p_array;

                if (x.length > 0 && y1.length > 0 && y2.length > 0) {
                    const ctx = document.getElementById('myChart').getContext('2d');
                    myChart = new Chart(ctx, {
                        type: 'line',
                        data: {
                            labels: x,
                            datasets: [
                                {
                                    label: 'OX Prediction (ppm)',
                                    data: y1,
                                    borderColor: 'rgb(75, 192, 192)',
                                    tension: 0.1,
                                    fill: false,
                                    yAxisID: 'y',
                                },
                                {
                                    label: 'Probability of exceeding 120ppm (%)',
                                    data: y2,
                                    borderColor: 'rgb(255, 99, 132)',
                                    tension: 0.1,
                                    fill: false,
                                    yAxisID: 'y-right',
                                },
                            ],
                        },
                        options: {
                            scales: {
                                x: {
                                    title: {
                                        display: true,
                                        text: 'Hours',
                                    },
                                },
                                y: {
                                    type: 'linear',
                                    position: 'left',
                                    beginAtZero: true,
                                    title: {
                                        display: true,
                                        text: 'OX (ppm)',
                                    },
                                },
                                'y-right': {
                                    type: 'linear',
                                    position: 'right',
                                    beginAtZero: true,
                                    title: {
                                        display: true,
                                        text: 'Probability (%)',
                                    },
                                    grid: {
                                        drawOnChartArea: false,
                                    },
                                },
                            },
                        },
                    });
                }
            }
        }
    }
</script>

<div id="map" style="height: 67vh; width: 100vw;">
  <div class="address-overlay">{address}</div>
</div>

<button on:click={moveToCurrentLocation}>現在地に移動</button><br />
地理院タイル: {X} {Y} (Zoomレベル12)<br />
起点時刻: {now}<br />
OX予測: {ox_array} ppm<br />
120 ppm越え確率(%): {p_max}<br />
<canvas id="myChart"></canvas>

<style>
    button {
        z-index: 10;
    }
    #map {
        position: relative; /* Make the map a positioning context */
    }

    .address-overlay {
        position: absolute; /* Absolute positioning within the map */
        top: 50%; /* Center vertically */
        left: 50%; /* Center horizontally */
        transform: translate(-50%, -50%); /* Adjust for element's size */
        background-color: rgba(255, 255, 255, 0.5); /* Semi-transparent white */
        padding: 4px;
        border-radius: 4px;
        border: 1px solid black;
        font-size: 12px;
        text-align: center; /* Center text */
        z-index: 1000; /* Ensure it's on top */
    }
</style>
