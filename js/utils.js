// === 通用工具函数 ===

function clamp(v, a, b) {
    return Math.max(a, Math.min(b, v));
}

function dist(a, b) {
    let dx = a.x - b.x, dy = a.y - b.y;
    return Math.hypot(dx, dy);
}

function angleTo(a, b) {
    return Math.atan2(b.y - a.y, b.x - a.x);
}

// 颜色映射配置
const intentColors = {
    // 玩家动作
    'player_left': '#FF6B6B', 'player_right': '#4ECDC4', 'player_up': '#45B7D1', 'player_down': '#FFA07A', 'player_stay': '#FFD93D',
    'player_attack': '#FF1744', 'player_dodge_left': '#9C27B0', 'player_dodge_right': '#7C3AED', 'player_dodge_up': '#673AB7', 'player_dodge_down': '#512DA8',
    'player_dodge_running': '#8E44AD',
    // Boss意图
    'boss_cooldown': '#95A5A6', 'boss_returnCenter': '#FF69B4', 'boss_escape': '#E74C3C', 'boss_farApproach': '#3498DB', 'boss_closeAttack': '#F39C12'
};