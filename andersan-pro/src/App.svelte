<script>
	import axios from 'axios';
	import { onMount } from 'svelte';

	let prefecture = 'kanagawa'; // デフォルトの都道府県
	let model = 'v0a'; // モデルを v0a に固定
	let oxData = [];
	let loading = false;
	let error = null;

	// 日時を ISO 8601 形式に変換する関数 (現在時刻用、JST に対応)
	function getCurrentISO8601() {
		const now = new Date();
		const jstOffset = 9 * 60; // JST は UTC+9 なので、9時間分のオフセットを分単位で設定
		const utc = now.getTime() + (now.getTimezoneOffset() * 60000); // UTC に変換
		const jst = new Date(utc + (jstOffset * 60000)); // JST に変換

		const year = jst.getFullYear();
		const month = (jst.getMonth() + 1).toString().padStart(2, '0'); // 月は 0-11 なので +1 する
		const day = jst.getDate().toString().padStart(2, '0');
		const hour = jst.getHours().toString().padStart(2, '0');
		const minute = jst.getMinutes().toString().padStart(2, '0');

		return `${year}-${month}-${day}T${hour}:${minute}:00+09:00`;
	}

	// データを取得する関数
	async function fetchData() {
		loading = true;
		error = null;
		oxData = [];

		try {
			const formattedDatehour = getCurrentISO8601(); // 現在時刻を取得 (JST)
			const response = await axios.get(
				// `http://172.23.78.71:8087/ox/${model}/${prefecture}/${formattedDatehour}`
				`http://192.168.3.234:8087/ox/${model}/${prefecture}/${formattedDatehour}`
			);

			// データの整形
			const all_data = response.data;
			const data = all_data.data;

			oxData = data; // data をそのまま oxData に代入

		} catch (err) {
			error = err.message || 'データの取得に失敗しました。';
		} finally {
			loading = false;
		}
	}

	// 初期データ取得
	onMount(fetchData);

	// 都道府県変更時の処理
	function handlePrefectureChange(event) {
		prefecture = event.target.value;
		fetchData();
	}
</script>

<main>
	<h1>OX 予測データ</h1>

	<div>
		<label for="prefecture">都道府県:</label>
		<select id="prefecture" bind:value={prefecture} on:change={handlePrefectureChange}>
			<option value="kanagawa">神奈川県</option>
			<!-- 他の都道府県もここに追加 -->
		</select>
	</div>

	{#if loading}
		<p>データを読み込み中です...</p>
	{:else if error}
		<p style="color: red;">エラー: {error}</p>
	{:else if oxData !== undefined}
		<table>
			<thead>
				<tr>
					<th>XY</th>
					<th>経度</th>
					<th>緯度</th>
					{#each Array.from({ length: 24 }, (_, i) => i + 1) as hour}
						<th>+{hour}</th>
					{/each}
				</tr>
			</thead>
			<tbody>
				{#each oxData.XY as item, i}
					<tr>
						<td>{oxData.XY[i]}</td>
						<td>{oxData.lon[i].toFixed(2)}</td>
						<td>{oxData.lat[i].toFixed(2)}</td>
						{#each Array.from({ length: 24 }, (_, j) => j + 1) as hour}
							<td style={Math.round(oxData[`+${hour}`][i]) > 40 ? 'color: red;' : ''}>
								{Math.round(oxData[`+${hour}`][i])}
							</td>
						{/each}
					</tr>
				{/each}
			</tbody>
		</table>
	{:else}
		<p>データがありません。</p>
	{/if}
</main>

<style>
	main {
		text-align: center;
		padding: 1em;
		max-width: 100%;
		margin: 0 auto;
	}

	table {
		width: 100%;
		border-collapse: collapse;
		margin-top: 1em;
	}

	th,
	td {
		border: 1px solid #ccc;
		padding: 0.5em;
		text-align: center;
	}
</style>
