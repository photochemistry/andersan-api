<script>
    import { onMount } from 'svelte';
    import 'leaflet/dist/leaflet.css';
    import L from 'leaflet';
     import { fetchData, fetchAddress, unixTimeToJSTString } from './retrieve.js';

    let map;
    let ox_dict;
    let address;
    let addr_dict;
    let ox_array;
    let now;
    let X, Y;
    // let longitude;
    // let latitude;

    function findMatchingRowIndex(array, targetValue1, targetValue2) {
        return array.findIndex(row => row[0] === targetValue1 && row[1] === targetValue2);
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
        // // Leafletのアイコンに関する設定（アイコンが表示されない問題の対策）
        // delete L.Icon.Default.prototype._getIconUrl;
        // L.Icon.Default.mergeOptions({
        //   iconRetinaUrl: require('leaflet/dist/images/marker-icon-2x.png'),
        //   iconUrl: require('leaflet/dist/images/marker-icon.png'),
        //   shadowUrl: require('leaflet/dist/images/marker-shadow.png'),
        // });

    });
  
    // 現在地を取得して地図を移動する関数
    const moveToCurrentLocation = () => {
        if (!navigator.geolocation) {
            alert('Geolocation is not supported by your browser');
            return;
        }
  
        const latitude = 35+20/60+8/3600;
        const longitude = 139+20/60+58/3600;
        // alert(longitude + " " + latitude)
        map.setView([latitude, longitude], 16); // 現在地に移動してズーム
        L.marker([latitude, longitude]).addTo(map); // 現在地にマーカーを追加
        fetchAddress(longitude, latitude).then(a=>{address=a.address; addr_dict = a;});
        fetchData().then(result=>{ox_dict=result});
        
        //   navigator.geolocation.getCurrentPosition(
        //     (position) => {
        //       const { latitude, longitude } = position.coords;
        //       map.setView([latitude, longitude], 16); // 現在地に移動してズーム
        //       L.marker([latitude, longitude]).addTo(map); // 現在地にマーカーを追加
        //     },
        //     () => {
        //       alert('Unable to retrieve your location');
        //     }
        //   );
    };

    $:{
        // 
        // console.log(ox.data);
        if (ox_dict !== undefined){
            if (addr_dict !== undefined){
                X = addr_dict.X;
                Y = addr_dict.Y;
                // alert(X+":"+Y)
                let row = findMatchingRowIndex(ox_dict.data.XY, X, Y);
                ox_array = [];
                for (let hr = 1; hr <= 24; hr++) {
                    ox_array.push(Math.round(ox_dict.data[`+${hr}`][row]));
                }
                now = unixTimeToJSTString(ox_dict.spec.timestamp[0]);
            }
        }
    }


</script>
  
<div id="map" style="height: 70vh; width: 100vw;"></div>

<button on:click={moveToCurrentLocation}>現在地に移動</button><br />
現在地: {address}付近<br />
地理院タイル: {X} {Y} (Zoomレベル12)<br />
起点時刻: {now}<br />
OX予測: {ox_array} ppm<br />
120 ppm越え確率: (未完成)

<style>
    button {
        z-index: 10;
    }
</style>