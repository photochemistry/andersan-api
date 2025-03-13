export async function fetchData() {
    try {
        const response = await fetch('http://172.23.78.65:8087/ox/v1a/kanagawa/2025-03-13T09:00+09:00'); // APIエンドポイントを指定
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json(); // レスポンスをJSON形式で取得
        console.log(data); // 取得したデータをコンソールに表示
        return data; // 取得したデータを返す
    } catch (error) {
        console.error('データの取得中にエラーが発生しました:', error);
        // エラーハンドリングを行う
    }
  }
  
//   fetchData()->then(result=>{console.log(result)}); // 関数を実行

export async function fetchAddress(lon, lat){
    http://172.23.78.65:8087/loc/139.34944444444446/35.33555555555556
    try {
        const response = await fetch(`http://172.23.78.65:8087/loc/${lon}/${lat}`); // APIエンドポイントを指定
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json(); // レスポンスをJSON形式で取得
        console.log(data); // 取得したデータをコンソールに表示
        return data; // 取得したデータを返す
    } catch (error) {
        console.error('データの取得中にエラーが発生しました:', error);
        // エラーハンドリングを行う
    }
}