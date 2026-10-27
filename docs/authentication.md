# 统一认证与执行器资源保护

本服务不维护人员密码库，不持有调度平台的 JWT 私钥。生产默认
`PERSON_SEARCH_AUTH_ENABLED=true`；缺少信任配置时受保护接口拒绝请求。
`auth_enabled=false` 仅用于显式隔离的开发/测试环境，不能作为生产兼容开关。

## 部署信任关系

每实例配置稳定且唯一的 `PERSON_SEARCH_EXECUTOR_ID`（默认 `executor-01`）：

```dotenv
PERSON_SEARCH_AUTH_ENABLED=true
PERSON_SEARCH_EXECUTOR_ID=executor-01
PERSON_SEARCH_AUTH_ISSUER=https://dispatch.example.invalid
PERSON_SEARCH_AUTH_JWKS_URL=https://dispatch.example.invalid/.well-known/jwks.json
PERSON_SEARCH_AUTH_STATUS_URL=https://dispatch.example.invalid/api/v1/auth/token-status
PERSON_SEARCH_AUTH_AUDIENCE=insightface:executor-01
# 私有 CA 按实际容器挂载配置；保持严格 TLS 验证。
# PERSON_SEARCH_AUTH_CA_FILE=/run/secrets/dispatch-ca.crt
```

JWKS 与状态地址只读取部署配置，强制 HTTPS，不跟随 Token 的 `jku` / `x5u`。
只接受 RS256、配置的签发方/受众、用户或服务类型及本实例身份；公钥缓存
5 分钟，未知 `kid` 受限刷新。公钥轮换须先发布新公钥，再切换签名，旧公钥保留至旧
Token 全部过期加时钟容差。状态接口携带原 Bearer Token，以 `{active:true}` 证明会话或
租约仍有效，缓存不超过 30 秒；状态接口故障拒绝新增访问并在下一次检查关闭流。

Token 声明包含 `iss/aud/exp/iat/sub/jti/type/executor_id/scopes`；用户另有 `sid`。
权限使用 `executor:read`、`executor:write`、`executor:cleanup`、`evidence:read`、`evidence:release`。
写操作必须带 `lease_id`。目标和搜索在内存中绑定创建租约，跨租约操作返回 403；
`replace_active` 也无法抢占其他租约。一个实例仍只有一个活动搜索，不能增加 Uvicorn worker。
多个可靠并发槽位需要多个经硬件资源验收的独立实例。

旧证据 API Key 配置、请求头和同步脚本已删除。证据只在启用统一认证时生成，读取和释放
必须使用绑定执行器、搜索租约及 `evidence:read` / `evidence:release` scope 的 JWT。
照片、视频和 embedding 不新增任何落盘行为。

## 浏览器监控与短期票据

生产管理入口为调度网关 `/insightface/<executor_id>/`。静态页面不含业务数据，
通过同源 `/api/v1/auth/csrf`、`/auth/refresh`、`/auth/exchange` 获得内存中的访问 JWT；
无登录会话时显示调度登录入口。HTTP 请求使用 Bearer，不写 Web Storage。

登记/启动之前，页面调用调度
`POST /api/v1/recognition-executors/<id>/leases`（`purpose=manual`），拿到租约专用 Token；
现有业务占槽时不能抢占。停止/结束后由调度的租约释放接口完成远端资源对账。
调度先将租约标为 releasing，再用 `type=service`、`executor:cleanup` 的专用 Token 调用
`DELETE /v1/leases/<lease_id>/resources`，停止本租约搜索、删除目标并核对资源释放。
执行器立即标记租约 closing，按租约串行化写入；即使旧写 Token 的状态缓存仍有效，
也不能在清理之后重新创建资源。失败保留关闭标记和资源归属，可重试清理，租约 ID 永不复用。

原生 img/WS 先使用 Bearer 调用：

```http
POST /v1/auth/tickets
Content-Type: application/json

{"path":"/v1/searches/<id>/events?after_seq=0","purpose":"websocket"}
```

返回 `{ticket,expires_in}`，最多 30 秒，单次消费，绑定完整路径和查询参数。
预览用 `purpose=preview`，只能绑定 `preview.mjpg`；WS 只能绑定 `/events`。
通过 `ticket` 查询参数或 WS `ticket.<ticket>` 子协议兑换，长期 JWT 不进入 URL。
流在父 Token 到期时关闭，最多 30 秒重新检查撤销；不会停止后台识别。
访问日志过滤票据，代理也须关闭含票据 URI 的日志和照片临时磁盘缓冲。

## 健康检查与发布验收

- `/livez` 是唯一匿名探针，只返回 `{"status":"ok"}`；Docker healthy 只代表进程存活。
- `/healthz`、`/readyz`、业务 API、文档、WS 和预览全部需要认证。
- `/healthz` 返回执行器身份、`capacity=1` 以及 JWT/租约/票据能力。
- 调度每次用新签发的执行器 JWT 检查 readiness 和 provider；不能用匿名 livez 代替。
- `scripts/check_rk3588_contract.py` 默认只检查 livez；设置
  `PERSON_SEARCH_PROBE_TOKEN_FILE` 为 owner-only 的临时 JWT 文件后，检查完整契约。
  `--require-ready` 在缺少该文件时失败。不要把 15 分钟 Token 固化为容器永久健康配置。
- T4 脚本要求 `T4_AUTH_ENV_FILE`（信任配置）和 `T4_PROBE_TOKEN_FILE`（新签发短期只读 JWT），
  后者只读挂载用于候选 readiness 验收。私有 CA 可用 `T4_AUTH_CA_FILE` 只读挂载。
  探针失败保留错误状态，不降级匿名检查。
- 原生 RK3588 发布启动服务时，要求 `RK3588_REMOTE_PROBE_TOKEN_FILE` 指向目标机上的
  owner-only 临时 JWT 文件；存活检查后执行强制 readiness 验收，通过才报告发布完成。

首次从无应用层认证版本切换须在维护窗口停止旧服务，禁止以匿名旧容器作为自动回退基线。

本轮仅本地实现与测试；现场 Compose 的设备、模型、host network/IPC、CA 挂载和
保护配置须在发布前逐项与实际运行容器核对。本轮未部署，真实硬件并发容量未验收。
